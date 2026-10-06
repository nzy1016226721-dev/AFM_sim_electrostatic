"""Aggregated imports for the postprocessing subpackage.

Provides a single access point for all postprocessing functions.
"""

from .plot_npy import plot_afm_from_npy
from .integrate_power import (
    energy_per_cycle, load_array, compute_power, slice_plot,
    line_plot, interactive_main as integrate_power_interactive
)
from .field_calculator import (
    compute_residual_Laplace, check_curl_E, check_dirichlet,
    check_neumann, cell_center_to_node, compute_current_divergence, integrate_power
)
from .sanity_check import run_sanity_check, plot_field_slice
from .field_lines import plot_field_lines, interactive_main as field_lines_interactive
from .potential_map import plot_potential_map, interactive_main as potential_map_interactive
from .npy_utils import parse_phi_filename


__all__ = [
    "plot_afm_from_npy",
    "energy_per_cycle", "load_array", "compute_power",
    "slice_plot", "line_plot", "integrate_power_interactive",
    "compute_residual_Laplace", "check_curl_E", "check_dirichlet",
    "check_neumann", "cell_center_to_node", "compute_current_divergence",
    "integrate_power", "run_sanity_check", "plot_field_slice",
    "plot_field_lines", "field_lines_interactive",
    "plot_potential_map", "potential_map_interactive",
    "parse_phi_filename",
]
