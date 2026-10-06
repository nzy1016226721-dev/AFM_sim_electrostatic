import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def load_curves(data):
    """Split paired-column data into individual (x, y) curves.

    Parameters
    ----------
    data : np.ndarray
        2D array where each pair of columns is one (x, y) curve.

    Returns
    -------
    list of np.ndarray
        Each element is a (N, 2) array with the curve points.
    """
    n_curves = data.shape[1] // 2
    curves = []
    for i in range(n_curves):
        x = data[:, 2*i]
        y = data[:, 2*i + 1]
        mask = ~np.isnan(x) & ~np.isnan(y)
        curve = np.column_stack((x[mask], y[mask]))
        curves.append(curve)
    return curves


def generate_eps_profile(epsilon_N_csv="epsilon_N.csv", As_z_csv="As_z.csv",
                         output_csv="eps_z.csv"):
    """Generate epsilon vs depth profile from material CSV data.

    Reads carrier concentration vs energy (epsilon_N) and doping vs depth
    (As_z), interpolates epsilon as a function of z, and saves the result.

    Parameters
    ----------
    epsilon_N_csv : str, optional
        Path to epsilon-N CSV (default: epsilon_N.csv).
    As_z_csv : str, optional
        Path to As-z doping CSV (default: As_z.csv).
    output_csv : str, optional
        Output path for the epsilon depth profile (default: eps_z.csv).

    Returns
    -------
    None
    """

    epsilon_N_data = np.genfromtxt(epsilon_N_csv, delimiter=",", skip_header=2)
    As_z_data = np.genfromtxt(As_z_csv, delimiter=",", skip_header=2)

    epsilon_N_curve = load_curves(epsilon_N_data)[0]
    As_z_curve = load_curves(As_z_data)[0]

    As_z_curve[:, 0] *= 1e-9
    As_z_curve[:, 1] *= 1e20

    epsilon_interpolated = interp1d(epsilon_N_curve[:, 0], epsilon_N_curve[:, 1],
                                    kind='cubic', bounds_error=False, fill_value=np.nan)
    As_interpolated = interp1d(As_z_curve[:, 0], As_z_curve[:, 1],
                               kind='cubic', bounds_error=False, fill_value=np.nan)

    z_m = np.linspace(0, 10e-9, 200)
    epsilon_profile = epsilon_interpolated(As_interpolated(z_m))

    output_data = np.column_stack((z_m, epsilon_profile))
    np.savetxt(output_csv, output_data, delimiter=",", header="z_nm,epsilon", comments="")

    print(f"Epsilon profile saved to {output_csv}")

    plt.figure(figsize=(8, 5))
    plt.plot(z_m, epsilon_profile)
    plt.xlabel(r'$z$ (m)')
    plt.ylabel(r'$\epsilon$')
    plt.title("Epsilon depth profile")
    plt.savefig("epsilon_profile.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    generate_eps_profile()
