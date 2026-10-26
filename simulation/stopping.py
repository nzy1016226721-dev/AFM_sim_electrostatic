"""Opt-in stopping decisions, independent of field storage and numerical kernels."""
from collections import deque
from copy import deepcopy
import json
import math
from pathlib import Path
import time

_ALLOWED = {"criterion", "rms_tol", "max_tol", "scope", "regions", "confirmation_checks",
            "history", "stagnation", "precision", "max_iterations"}
_MODES = ("off", "observe", "stop")


def _number(value, name, positive=True):
    if isinstance(value, bool):
        raise ValueError(name + " must be numeric")
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(name + " must be finite and " + ("positive" if positive else "nonnegative"))
    return value


def _integer(value, name, minimum=1):
    result = _number(value, name, minimum > 0)
    if result != int(result) or result < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return int(result)


def _object(value, name, allowed):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(name + " has unknown keys or is not an object")
    return deepcopy(value)


def resolve_stopping(spec, tol):
    """Validate without NumPy/JIT imports; None preserves the legacy path."""
    if spec is None:
        return None
    out = _object(spec, "stopping", _ALLOWED)
    out.setdefault("criterion", "rms")
    if out["criterion"] not in ("rms", "maximum", "both"):
        raise ValueError("stopping.criterion must be rms, maximum or both")
    out["rms_tol"] = _number(out.get("rms_tol", tol), "stopping.rms_tol")
    out["max_tol"] = _number(out.get("max_tol", tol), "stopping.max_tol")
    out.setdefault("scope", "finest")
    if out["scope"] not in ("finest", "all_levels"):
        raise ValueError("stopping.scope must be finest or all_levels")
    out["confirmation_checks"] = _integer(out.get("confirmation_checks", 1), "confirmation_checks")
    regions = _object(out.get("regions", {}), "stopping.regions", ("cut", "qd"))
    for name, options in regions.items():
        options = _object(options, name, ("rms_tol", "max_tol"))
        if not options:
            raise ValueError(name + " must request at least one residual norm")
        regions[name] = {key: _number(value, name + "." + key) for key, value in options.items()}
    out["regions"] = regions
    history = _object(out.get("history", {}), "stopping.history",
                      ("mode", "window", "mean_tol", "max_tol", "probe_updates"))
    history.setdefault("mode", "off")
    if history["mode"] not in ("off", "observe", "require"):
        raise ValueError("history.mode must be off, observe or require")
    history["window"] = _integer(history.get("window", 3), "history.window", 2)
    history["probe_updates"] = _integer(history.get("probe_updates", 10), "history.probe_updates")
    for key in ("mean_tol", "max_tol"):
        if key in history:
            history[key] = _number(history[key], "history." + key)
    if history["mode"] == "require" and not all(key in history for key in ("mean_tol", "max_tol")):
        raise ValueError("Required history needs explicit calibrated mean_tol and max_tol")
    out["history"] = history
    for name in ("stagnation", "precision"):
        options = _object(out.get(name, {}), "stopping." + name,
                          ("mode", "window", "relative_improvement", "motion_tol"))
        options.setdefault("mode", "off")
        if options["mode"] not in _MODES:
            raise ValueError(name + ".mode must be off, observe or stop")
        options["window"] = _integer(options.get("window", 5), name + ".window", 3)
        # Observation can be instrumented before the acceptance constants are frozen.
        for key in ("relative_improvement", "motion_tol"):
            if key in options:
                options[key] = _number(options[key], name + "." + key, positive=False)
        if options["mode"] == "stop" and not all(
                key in options for key in ("relative_improvement", "motion_tol")):
            raise ValueError(name + " early stop needs explicit calibrated thresholds")
        out[name] = options
    if out.get("max_iterations") is not None:
        out["max_iterations"] = _integer(out["max_iterations"], "stopping.max_iterations")
    return out


def level_stopping(spec, shape, target, tol):
    value = resolve_stopping(spec, tol)
    return value if value is not None and (value["scope"] == "all_levels" or tuple(shape) == tuple(target)) else None


def trial_output_filename(filename, spec, level_reports):
    """Label opt-in unqualified output without changing legacy file contracts."""
    if spec is None or (level_reports and all(row.get("reason")=="converged" for row in level_reports)):
        return filename
    return filename if filename.startswith("trial_") else "trial_"+filename


def _finite_copy(value):
    if isinstance(value, dict):
        return {key: _finite_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_copy(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class StoppingController:
    """One level's bounded scalar history and identical shared/MPI decisions."""

    def __init__(self, spec, tol, shape, context=None, output_dir=None, writer=True):
        self.spec = resolve_stopping(spec, tol)
        if self.spec is None:
            raise ValueError("Instantiate only for an opt-in stopping policy")
        self.shape = tuple(int(v) for v in shape)
        self.context = deepcopy(context or {})
        self.context["drive_scale_volts"] = _number(self.context.get("drive_scale_volts",1.0),"drive_scale_volts")
        self.timings = {"update_restore_seconds":0.0,"halo_seconds":0.0,"global_residual_seconds":0.0}
        length = max(self.spec["history"]["window"], self.spec["stagnation"]["window"],
                     self.spec["precision"]["window"])
        self.history = deque(maxlen=length)
        self.passing = 0
        self.candidate = False
        self.probe_until = 0
        self.probe_started = False
        self.last = None
        self.last_count = None
        self.history_gap = None
        self.checks = 0
        self.extra_seconds = 0.0
        self.output_dir = Path(output_dir) if writer and output_dir is not None else None
        self.started = time.monotonic()

    @property
    def needs_precision(self):
        return self.spec["precision"]["mode"] != "off"

    def needs_check(self, iteration, normal):
        return normal or self.candidate or iteration <= self.probe_until

    def limited(self, iteration):
        limit = self.spec.get("max_iterations")
        return limit is not None and iteration >= limit

    def _magnitudes(self, stats):
        norms = []
        criterion = self.spec["criterion"]
        if criterion in ("rms", "both"):
            norms.append(("global.rms", stats["global"]["rms"], self.spec["rms_tol"]))
        if criterion in ("maximum", "both"):
            norms.append(("global.maximum", stats["global"]["maximum"], self.spec["max_tol"]))
        for name, options in self.spec["regions"].items():
            for key, threshold in options.items():
                metric = "rms" if key == "rms_tol" else "maximum"
                norms.append((name + "." + metric, stats[name][metric], threshold))
        return norms

    def observe(self, iteration, rms, maximum, count, extras=None):
        stats = {"global": {"rms": float(rms), "maximum": float(maximum), "count": int(count)}}
        for name, value in (extras or {}).items():
            if name == "global":
                stats["global"].update({key: item for key, item in value.items()
                                        if key not in ("rms", "maximum", "count")})
            else:
                stats[name] = value
        for item in stats.values():
            item["physical_native_rms"] = item["rms"] * self.context["drive_scale_volts"]
            item["physical_native_maximum"] = item["maximum"] * self.context["drive_scale_volts"]
        self.checks += 1
        required = ("global", *self.spec["regions"])
        reason = "running"
        if any(name not in stats or int(stats[name]["count"]) <= 0 for name in required):
            reason = "invalid_region"
        elif any(not math.isfinite(float(stats[name][key])) for name in required for key in ("rms", "maximum")):
            reason = "invalid_numerics"
        count_identity = tuple((name, int(stats.get(name, {}).get("count", 0))) for name in required)
        if count_identity != self.last_count:
            self.history.clear()
            self.history_gap = None
            self.passing = 0
            self.last_count = count_identity
        norms = self._magnitudes(stats) if reason == "running" else []
        magnitudes_pass = bool(norms) and all(value < threshold for _, value, threshold in norms)
        if self.history:
            gap = iteration - self.history[-1]["iteration"]
            if gap <= 0:
                raise ValueError("Stopping observations must have increasing iteration numbers")
            if self.history_gap is not None and gap != self.history_gap:
                # A uniform10-update diagnostic window is valid, but must not
                # be confused with the one-update confirmation/motion window.
                previous = self.history[-1]
                self.history.clear()
                self.history.append(previous)
            self.history_gap = gap
        values = {key: value for key, value, _ in norms}
        self.history.append({"iteration": int(iteration), "values": values})
        motion = {}
        window = self.spec["history"]["window"]
        if len(self.history) >= window:
            records = list(self.history)[-window:]
            for key in values:
                changes = [abs(b["values"][key] - a["values"][key]) for a, b in zip(records, records[1:])]
                motion[key] = {"mean_absolute_change": sum(changes) / len(changes),
                               "maximum_absolute_change": max(changes),
                               "iteration_gap": self.history_gap}
        history_spec = self.spec["history"]
        history_pass = history_spec["mode"] != "require" or (
            bool(motion) and self.history_gap == 1 and all(value["mean_absolute_change"] < history_spec["mean_tol"] and
                                 value["maximum_absolute_change"] < history_spec["max_tol"]
                                 for value in motion.values()))
        if magnitudes_pass:
            self.passing += 1
            self.candidate = True
            if self.passing >= self.spec["confirmation_checks"] and history_pass:
                reason = "converged"
        else:
            self.passing = 0
            self.candidate = False
        # A single bounded dense-check probe collects pilot data even if a
        # stricter maximum prevents entry into successful-pass confirmation.
        observe_enabled = (history_spec["mode"] != "off" or self.spec["stagnation"]["mode"] != "off"
                           or self.spec["precision"]["mode"] != "off")
        if observe_enabled and not self.probe_started and math.isfinite(float(rms)) and rms < 2 * self.spec["rms_tol"]:
            self.probe_until = iteration + history_spec["probe_updates"]
            self.probe_started = True
        plateaus = {}
        for detector in ("precision", "stagnation"):
            options = self.spec[detector]
            n = options["window"]
            if options["mode"] == "off" or len(self.history) < n or not norms:
                continue
            samples = list(self.history)[-n:]
            failing = [(key, value, threshold) for key, value, threshold in norms if value >= threshold]
            details = {}
            for key, _, _ in failing:
                series = [row["values"][key] for row in samples]
                improvement = (series[0] - min(series)) / max(abs(series[0]), 1e-300)
                changes = [abs(b - a) for a, b in zip(series, series[1:])]
                details[key] = {"relative_best_improvement": improvement, "maximum_change": max(changes),
                                "iteration_gap": self.history_gap}
            plateaus[detector] = details
            calibrated = "relative_improvement" in options and "motion_tol" in options
            flat = bool(details) and calibrated and all(
                item["relative_best_improvement"] <= options["relative_improvement"] and
                item["maximum_change"] <= options["motion_tol"] for item in details.values())
            if reason == "running" and flat and options["mode"] == "stop":
                if detector == "stagnation":
                    reason = "stagnated"
                else:
                    explained = all(
                        "precision" in stats[key.split(".")[0]] and
                        stats[key.split(".")[0]]["precision"][key.split(".")[1]] < threshold
                        for key, value, threshold in failing)
                    if explained:
                        reason = "precision_limited"
        if reason == "running" and self.limited(iteration):
            reason = "iteration_limit"
        self.last = {"iteration": int(iteration), "shape": self.shape, "reason": reason,
                     "statistics": stats, "magnitudes_pass": magnitudes_pass, "history_pass": history_pass,
                     "consecutive_magnitude_passes": self.passing, "norm_motion": motion,
                     "plateau_observations": plateaus, "case_tag": self.context.get("case_tag"),
                     "drive_scale_volts": self.context.get("drive_scale_volts", 1.0)}
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "stopping_history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_finite_copy(self.last), allow_nan=False) + "\n")
        return reason

    def summary(self):
        return _finite_copy({"policy": self.spec, "checks": self.checks, "last": self.last,
                             "timings":self.timings,
                             "extra_statistics_seconds": self.extra_seconds,
                             "full_residual_history_bytes": 0, "scalar_history_snapshots": len(self.history)})

