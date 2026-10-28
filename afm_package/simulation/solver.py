import numpy as np
import time
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

MG_TIME = {"elapsed": 0.0}
RESIDUAL_CSV = "residual_history.csv"


def build_downward_pointing_tip(nx, ny, nz, tip_z=0.2, R=0.05, r_tip=0.15,
                                aspect_ratio=2.0, verbose=True):
    """
    Generates a boolean mask for a hyperbolic AFM tip geometry.

    The tip is modelled as a hyperboloid pointing downward along z,
    centred at (0.5, 0.5) in xy, with curvature radius R, truncation
    radius r_tip, and vertical aspect ratio.

    """

    x = np.linspace(0, 1, nx)  #Defines three 1D arrays with fractional coordinates
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    cx, cy = 0.5, 0.5  #Defines (in fractional coordinates) the location of the center of the tip
    
    tip_z_idx = int(round(tip_z * (nz - 1))) #Takes the fractional z-height and transforms it into grid units
    z_tip = z[tip_z_idx]  #The fractional height of the lowest point on the tip
    #Make a empty 3D boolean array of False to replace with True for all (x,y,z) locations that make up the tip
    mask = np.zeros((nx, ny, nz), dtype=bool) 

    X, Y = np.meshgrid(x, y, indexing = 'ij') #Prepares a 2D matrix for computing all the radial distances
    r = np.hypot(X-cx, Y-cy) #Creates a matrix of values defined by their distances from the center 
    
    #Defines the mathematical equations necessary to build the tip
    theta_asym = np.arctan(aspect_ratio) #A bit redundant, as tan(arctan(x))=x
    a = R * np.tan(theta_asym)  #R is the curvature at the apex, multiplied by the aspect ratio
    b = R * np.tan(theta_asym) ** 2

    z0 = z_tip - b #The center of the hyperboloid
    z_base = z0 + np.sqrt(b**2 * (1 + (r_tip**2 / a**2))) #The hyperboloid equation evaluated at r_tip
    #i.e. z_base is the height of the tip at its widest point. I usually set r_tip to be so great that the tip extends
    #  out of the simulation space, so the tip will not be accidentally truncated

    for k in range(nz): #Loop through all z-heights
        zk = z[k]
        if zk < z_tip or zk > z_base: #Only continue if the z-height is within the bounds of where the tip is defined
            continue
        dz = zk - z0

        r_max = a * np.sqrt((dz**2 / b**2) - 1)  #Radius of the tip at height dz
        r_max = min(r_max, r_tip)  #r_tip is the maximum allowed radius, the point at which the build is truncated
       #For each z height set the mask to true for all (x,y) values of the distances mesh 
       #  that are less than the radius of the tip at this height, i.e. within the tip.
        mask[:, :, k] = r <= r_max 
                                    
        
    return mask, z_tip, z_base


def compute_residual_vec_unpadded(V, mask, axp, axm, ayp, aym, azp, azm, a0):
    """Compute residual of the discretized Poisson equation (full matrix output).

    Evaluates L(phi) = div(eps * grad(phi)) at interior points, masks
    boundary voxels to NaN, and returns the full residual matrix alongside
    summary statistics.

    Parameters
    ----------
    V : np.ndarray
        3D potential array (Nx x Ny x Nz).
    mask : np.ndarray (bool)
        Boundary mask (True = Dirichlet nodes).
    axp, axm, ayp, aym, azp, azm : np.ndarray
        Harmonic-mean dielectric coefficients at cell faces.
    a0 : np.ndarray
        Sum of the six neighbour coefficients.

    Returns
    -------
    res_rms : float
        RMS residual over interior nodes.
    res_max : float
        Maximum absolute residual over interior nodes.
    res_matrix : np.ndarray
        3D array of |residual| with NaN at boundaries.
    """
    #Nx, Ny, Nz = V.shape  

    num = (
        axp * V[2:  , 1:-1, 1:-1] +
        axm * V[ :-2, 1:-1, 1:-1] +
        ayp * V[1:-1, 2:  , 1:-1] +
        aym * V[1:-1,  :-2, 1:-1] +
        azp * V[1:-1, 1:-1, 2:  ] +
        azm * V[1:-1, 1:-1,  :-2]
    )
    #center = a0 * V[1:-1, 1:-1, 1:-1]
    resid_core = (num/a0) - V[1:-1, 1:-1, 1:-1]

    interior_mask = ~mask[1:-1, 1:-1, 1:-1]
    resid_core_masked = np.zeros_like(resid_core)
    resid_core_masked[~interior_mask] = np.nan
    resid_core_masked[interior_mask] = resid_core[interior_mask]

    if np.any(interior_mask):
        res_rms = np.sqrt(np.mean(resid_core[interior_mask]**2))
        res_max  = np.max(np.abs(resid_core[interior_mask]))
    else:
        res_rms, res_max = 0.0, 0.0

    res_matrix = np.full_like(V, np.nan, dtype=float)
    res_matrix[1:-1, 1:-1, 1:-1] = np.abs(resid_core_masked)

    return res_rms, res_max, res_matrix


def compute_residual_scalars(V, mask, axp, axm, ayp, aym, azp, azm, a0):
    """Compute scalar residual norms (no full matrix allocation).

    Parameters
    ----------
    V : np.ndarray
        3D potential array.
    mask : np.ndarray (bool)
        Boundary mask.
    axp, axm, ayp, aym, azp, azm : np.ndarray
        Dielectric face coefficients.
    a0 : np.ndarray
        Sum of coefficients.

    Returns
    -------
    res_L2 : float
        L2-norm residual.
    res_max : float
        Maximum residual.
    """
    num = (
        axp * V[2:  , 1:-1, 1:-1] +
        axm * V[ :-2, 1:-1, 1:-1] +
        ayp * V[1:-1, 2:  , 1:-1] +
        aym * V[1:-1,  :-2, 1:-1] +
        azp * V[1:-1, 1:-1, 2:  ] +
        azm * V[1:-1, 1:-1,  :-2]
    )
    resid = (num/a0) - V[1:-1, 1:-1, 1:-1]

    free = ~mask[1:-1, 1:-1, 1:-1]
    if not free.any():
        return 0.0, 0.0
    res_L2  = np.sqrt(np.mean(resid[free]**2))  
    #The L2 norm provides greater weighting to the cells that have a larger residual
    #res_max = np.sqrt(np.max(np.abs(resid[free])**2))
    res_max = np.max(np.abs(resid[free]))
    return res_L2, res_max




def log_residual_csv(iteration, res_avg, res_max, csv_file=None, output_dir="."):
    if csv_file is None:
        csv_file = RESIDUAL_CSV
    csv_path = os.path.join(output_dir, csv_file)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["iteration", "residual_avg", "residual_max"])
        writer.writerow([iteration, res_avg, res_max])


def plot_convergence(csv_file=None, output_dir=".", title=None):
    """
    Plot convergence history from residual CSV.
    If csv_file is None, use RESIDUAL_CSV.
    """
    if csv_file is None:
        csv_file = RESIDUAL_CSV

    # Prepend output directory if path is relative
    if not os.path.isabs(csv_file):
        csv_file = os.path.join(output_dir, csv_file)

    if not os.path.isfile(csv_file):
        print(f"Warning: residual file not found: {csv_file}")
        return

    # Try to read with different encodings
    encodings = ['utf-8-sig', 'latin-1', 'cp1252']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_file, encoding=enc)
            break
        except (UnicodeDecodeError, pd.errors.EmptyDataError):
            continue
    if df is None:
        print(f"Warning: could not read {csv_file} with any encoding. Skipping plot.")
        return

    if df.empty:
        print("Warning: residual file is empty. Skipping plot.")
        return

    fig = plt.figure(figsize=(7, 5))
    plt.semilogy(df["iteration"], df["residual_avg"],
                 marker="o", markersize=2, linestyle="None", color="tab:blue",
                 label="Average (L2) residual")
    plt.semilogy(df["iteration"], df["residual_max"],
                 marker="x", markersize=2, linestyle="None", color="tab:red",
                 label="Max residual")
    plt.xlabel("Iteration")
    plt.ylabel("Residual (V)")
    plt.title("Convergence History (RMS vs Max)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def mg_3d_masked(Vtip, phi, boundary_mask, damping=0.857,
                 max_iter=50, tol=1e-3, max_tol=1e-5, verbose=True, eps_r=None, eps=True,
                 mg_max_runtime=None, output_dir="."):
    """Multigrid solver for the 3D Poisson equation with dielectric variation.

    Uses successive over-relaxation (SOR) with Neumann boundary conditions
    on free faces and Dirichlet on tip/gate boundaries. Implements two
    inner solver variants (legacy and optimised).

    Parameters
    ----------
    Vtip : float
        Tip voltage (V).
    phi : np.ndarray
        Initial potential array (Nx x Ny x Nz).
    boundary_mask : np.ndarray (bool)
        True at Dirichlet boundary nodes.
    damping : float, optional
        SOR damping factor / omega (default: 0.857).
    max_iter : int, optional
        Maximum solver iterations (default: 50).
    tol : float, optional
        Convergence tolerance on L2 residual (default: 1e-6).
    verbose : bool, optional
        If True, print iteration info (default: True).
    eps_r : np.ndarray or None, optional
        Dielectric constant cell array (Nx-1 x Ny-1 x Nz-1).
    eps : bool, optional
        If True, solve with dielectric (default: True).
    mg_max_runtime : float or None, optional
        Max wall-clock time in seconds (default: None).
    output_dir : str, optional
        Directory for log files (default: ".").

    Returns
    -------
    phi : np.ndarray
        Solved potential array.
    residual : np.ndarray
        Final residual matrix (NaN at boundaries).
    """
    nx, ny, nz = phi.shape

    def neumann(a):
        """Apply homogeneous Neumann BC by copying the nearest interior plane."""
        a[0,:,:] = a[1,:,:]; a[-1,:,:] = a[-2,:,:]
        a[:,0,:] = a[:,1,:]; a[:,-1,:] = a[:,-2,:]
        a[:,:,0] = a[:,:,1]; a[:,:,-1] = a[:,:,-2]
        return a

    print(f"   Starting 3D MG solver: {nx}x{ny}x{nz} grid")



    def solve_varying_dielectric_3d_zero_rhs(V, eps_cell, max_iter, 
                                             omega=0.857, tol=1e-3, max_tol=1e-5,
                                              mask=None, verbose=False,
                                              output_dir="."):
        """Optimised SOR solver for div(eps * grad(phi)) = 0."""
        #NOTE: It appears to be an incorrect comment above, this is not an SOR solver as I understand it
        
        conv = False  #boolean variable set now, returned in filename, boolean converted to string in batch_main()
        
        Nx, Ny, Nz = V.shape #The current grid size for the iterative process about to be performed
        
        if eps_cell.shape != (Nx-1, Ny-1, Nz-1):
            raise ValueError("eps_cell must have shape (Nx-1,Ny-1,Nz-1).")

        if mask is None:
            mask = np.zeros_like(V, dtype=bool)

        Vfix = V.copy()

        ec = eps_cell.astype(np.float32, copy=False)
        axp = 0.25 * (ec[1:, :-1, :-1] + ec[1:, 1:, :-1] + ec[1:, :-1, 1:] + ec[1:, 1:, 1:])
        axm = 0.25 * (ec[:-1, :-1, :-1] + ec[:-1, 1:, :-1] + ec[:-1, :-1, 1:] + ec[:-1, 1:, 1:])
        ayp = 0.25 * (ec[:-1, 1:, :-1] + ec[1:, 1:, :-1] + ec[:-1, 1:, 1:] + ec[1:, 1:, 1:])
        aym = 0.25 * (ec[:-1, :-1, :-1] + ec[1:, :-1, :-1] + ec[:-1, :-1, 1:] + ec[1:, :-1, 1:])
        azp = 0.25 * (ec[:-1, :-1, 1:] + ec[1:, :-1, 1:] + ec[:-1, 1:, 1:] + ec[1:, 1:, 1:])
        azm = 0.25 * (ec[:-1, :-1, :-1] + ec[1:, :-1, :-1] + ec[:-1, 1:, :-1] + ec[1:, 1:, :-1])

        a0 = axp + axm + ayp + aym + azp + azm

        # ---------- Ensure residual CSV exists with header ----------
        # csv_path = os.path.join(output_dir, RESIDUAL_CSV)
        # if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        #     with open(csv_path, 'w', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(["iteration", "residual_avg", "residual_max"])
        # -----------------------------------------------------------

        csv_path = os.path.join(output_dir, RESIDUAL_CSV)
        with open(csv_path, 'w', newline='') as f:   # 'w' truncates: one solve = one history
            csv.writer(f).writerow(["iteration", "residual_avg", "residual_max"])

    
        Ni, Nj, Nk = Nx-2, Ny-2, Nz-2

        num       = np.empty((Ni, Nj, Nk), dtype=np.float32)
        R_arr     = np.empty((Ni, Nj, Nk), dtype=np.float32)

        V_int = V[1:-1, 1:-1, 1:-1]
        V_xp = V[2:  , 1:-1, 1:-1]
        V_xm = V[ :-2, 1:-1, 1:-1]
        V_yp = V[1:-1, 2:  , 1:-1]
        V_ym = V[1:-1,  :-2, 1:-1]
        V_zp = V[1:-1, 1:-1, 2:  ]
        V_zm = V[1:-1, 1:-1,  :-2]

        level_start_time = time.time()


        for it in range(1, max_iter + 1):
            np.multiply(axp, V_xp, out=num)
            for a, Vn in ((axm, V_xm), (ayp, V_yp), (aym, V_ym), (azp, V_zp), (azm, V_zm)):
                np.multiply(a, Vn, out=R_arr)
                np.add(num, R_arr, out=num)

            np.divide(num, a0, out=num)
            np.subtract(num, V_int, out=R_arr) 
            
            
            #R_arr is num/a0 - V, where num is the sum of all ax*Vx, ay*Vy, etc. 
            #Now it performs the Jacobi smoothing, omega*residual is the correction amount to V_int
            np.multiply(omega, R_arr, out=num) #re-using scratch again
            np.add(num, V_int, out=V_int)
            
            neumann(V)
            V[mask] = Vfix[mask]

            elapsed = time.time() - level_start_time
            MG_TIME["elapsed"] = elapsed


            if it % 10 == 0 or it <= 5:
                resL2, res_max = compute_residual_scalars(
                    V, mask, axp, axm, ayp, aym, azp, azm, a0)
  
                if verbose:
                    print(f"iter {it:6d}: res_avg={resL2:.5e}, res_max={res_max:.5e}")
                log_residual_csv(it, resL2, res_max, output_dir=output_dir)

                if resL2 < tol and res_max < max_tol:
                    if verbose:
                        print(f"Converged in {it} iterations; rms residual={resL2:.5e}, global max residual={res_max:.5e}")
                    plot_convergence(output_dir=output_dir, title=f"{Nx}x{Ny}x{Nz} grid")
                    conv = True
                    print(f'Number of iterations performed: {it}')
                    print(f'resL2 = {resL2}')
                    print(f'global res_max = {res_max}')
                   
                    #And for plotting residuals get the matrix out
                    resL2, res_max, res_matrix = compute_residual_vec_unpadded(V, mask, axp, axm, ayp, aym, azp, azm, a0)
                    return V, it, resL2, res_max, res_matrix, conv 


            if mg_max_runtime is not None and elapsed > mg_max_runtime:
                print(f"MG aborted early: exceeded max runtime ({mg_max_runtime} s). "
                      f"Completed {it} iterations.")
                plot_convergence(output_dir=output_dir, title=f"{Nx}x{Ny}x{Nz} grid")
                resL2, res_max, res_matrix = compute_residual_vec_unpadded(V, mask, axp, axm, ayp, aym, azp, azm, a0)
                full_res = np.full_like(V, np.nan, dtype=np.float32)
                full_res[1:-1, 1:-1, 1:-1] = np.abs(R_arr)
                return V, it, resL2, res_max, res_matrix, conv
            

                
            
        # If we exit the loop without converging or hitting runtime limit

        resL2, res_max, res_matrix = compute_residual_vec_unpadded(V, mask, axp, axm, ayp, aym, azp, azm, a0)
        print(f"NOT converged after {max_iter} iterations; last residual={resL2:.5e}")
        plot_convergence(output_dir=output_dir, title=f"{Nx}x{Ny}x{Nz} grid")
        full_res = np.full_like(V, np.nan, dtype=np.float32)
        full_res[1:-1, 1:-1, 1:-1] = np.abs(R_arr)
        return V, it, resL2, res_max, res_matrix, conv

    # ---- End of inner solver definition ----

    if not eps:
        eps_r = np.ones((nx - 1, ny - 1, nz - 1), dtype=np.float32)
    return solve_varying_dielectric_3d_zero_rhs(
            phi, eps_r, max_iter, damping, tol, max_tol, boundary_mask, verbose,
            output_dir=output_dir
        )