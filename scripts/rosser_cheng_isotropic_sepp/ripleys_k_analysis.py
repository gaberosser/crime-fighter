import numpy as np
from stats import ripley
from analysis.spatial import shapely_rectangle_from_vertices



def run_anisotropic_k(data,
                      boundary,
                      dmax=400,
                      nsim=50):
    # default behaviour; start at pi/8 and move in intervals of pi/4
    rk = ripley.RipleyKAnisotropicC2(data, dmax, boundary)
    rk.process()

    u = np.linspace(0, dmax, 200)
    k_obs = rk.compute_k(u)
    if not nsim:
        return u, k_obs
    k_sim = rk.run_permutation(u, niter=nsim)
    return u, k_obs, k_sim