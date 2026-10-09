"""Original snapshot Jacobi with compact integer-index material/mask adapters."""
import math
import time
import numpy as np
from . import ram_first as ram
from .compact_storage import PlaneEpsilon, PackedMask, restore_fixed
from .parallel import configure_cpu_threads


def solve_compact(phi, epsilon, mask, *, damping, tol, cpu_threads, max_runtime,
                  output_dir, verbose=False, diagnostic_iterations=None,
                  phi_update_mode=None, residual_accumulation=None):
    from .solver import MG_TIME, log_residual_csv
    if phi.dtype != np.float32 or epsilon.dtype != np.float32:
        raise ValueError("ram_compact requires float32 potentials and epsilon")
    if epsilon.shape != tuple(v-1 for v in phi.shape) or mask.shape != phi.shape:
        raise ValueError("Compact material/mask shapes must match the original grid")
    owned_epsilon = not isinstance(epsilon,PlaneEpsilon)
    epsilon = PlaneEpsilon.from_dense(epsilon, directory=output_dir) if owned_epsilon else epsilon
    try:
        mask = mask if isinstance(mask,PackedMask) else PackedMask.from_dense(mask)
        threads = configure_cpu_threads(cpu_threads)
        update_mode, accumulation = ram.resolve_storage_options("ram_compact",phi_update_mode,residual_accumulation)
        owned_shape = tuple(n-2 for n in phi.shape)
        mask_view = mask.kernel_view(offset=(1,1,1),shape=owned_shape)
        scratch = np.empty((min(threads,owned_shape[0]),3,phi.shape[1],phi.shape[2]),np.float32)
        target = np.empty_like(phi) if update_mode == "buffered" else None
        residual_volume = np.empty(owned_shape,np.float32) if accumulation == "array" else None
        fixed, prefixes = mask.fixed_values(phi)
        if diagnostic_iterations is not None and int(diagnostic_iterations) < 1:
            raise ValueError("diagnostic_iterations must be positive")
        cells = epsilon.kernel
        started = time.monotonic()
        iteration = 0
        while True:
            iteration += 1
            destination = phi
            if target is not None:
                np.copyto(target,phi)
                destination = target
            ram.jacobi_cells_in_place(destination,cells,mask_view,(1,1,1),phi.shape,
                                     np.float32(damping),scratch,True)
            destination[0,:,:] = destination[1,:,:]; destination[-1,:,:] = destination[-2,:,:]
            destination[:,0,:] = destination[:,1,:]; destination[:,-1,:] = destination[:,-2,:]
            destination[:,:,0] = destination[:,:,1]; destination[:,:,-1] = destination[:,:,-2]
            restore_fixed(destination,mask.bits,mask.shape,fixed,prefixes,(0,0,0))
            if target is not None:
                phi, target = destination, phi
            elapsed = time.monotonic()-started
            check = iteration<=5 or iteration%10==0
            diagnostic_end = diagnostic_iterations is not None and iteration>=int(diagnostic_iterations)
            timeout = max_runtime is not None and elapsed>float(max_runtime)
            if check or diagnostic_end or timeout:
                total, maximum, count = ram.residual_cells_stats(phi,cells,mask_view,(1,1,1),phi.shape,
                                                               accumulation,residual_volume)
                rms = math.sqrt(total/count) if count else 0.
                log_residual_csv(iteration,rms,maximum,output_dir=output_dir)
                if verbose:
                    print(f"iter {iteration:6d}: res_avg={rms:.5e}, res_max={maximum:.5e}")
            reason = "running"
            if diagnostic_iterations is None and check and rms<float(tol):
                reason = "converged"
            elif diagnostic_end:
                reason = "diagnostic_iteration_limit"
            elif timeout:
                reason = "time_limit"
            MG_TIME.update(elapsed=elapsed,iterations=iteration,residual=float(rms),residual_max=float(maximum),
                reason=reason,memory_mode="ram_compact",scratch_bytes=int(scratch.nbytes),
                phi_update_mode=update_mode,residual_accumulation=accumulation,
                phi_buffer_bytes=0 if target is None else int(target.nbytes),
                residual_workspace_bytes=0 if residual_volume is None else int(residual_volume.nbytes),
                epsilon_storage_bytes=epsilon.nbytes,epsilon_dense_bytes=math.prod(epsilon.shape)*4,
                epsilon_unique_planes=int(epsilon.bank.shape[0]),mask_storage_bytes=mask.nbytes,
                mask_dense_bytes=math.prod(mask.shape),fixed_values_bytes=int(fixed.nbytes),
                fixed_prefix_bytes=int(prefixes.nbytes))
            if reason != "running":
                print(f"RAM-compact {reason}: {iteration} iterations in {elapsed:.2f} s; residual={rms:.5e}")
                return phi,None
    finally:
        if owned_epsilon:
            epsilon.close()
