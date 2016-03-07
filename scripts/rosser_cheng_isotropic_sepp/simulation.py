import numpy as np
from network import simulate, itn
from point_process.simulate import NetworkHomogBgExponentialGaussianTrig
from stats import ripley
from analysis.spatial import shapely_rectangle_from_vertices
import os
import dill
from scripts import OUT_DIR


def run_anisotropic_k(net_pts,
                      dmax=400,
                      nsim=50):
    bds = net_pts.graph.extent
    xr = bds[2] - bds[0]
    yr = bds[3] - bds[1]
    buff = 0.01
    bd_poly = shapely_rectangle_from_vertices(
        bds[0] - xr * buff,
        bds[1] - yr * buff,
        bds[2] + xr * buff,
        bds[3] + yr * buff,
    )
    xy = net_pts.to_cartesian()
    # default behaviour; start at pi/8 and move in intervals of pi/4
    rk = ripley.RipleyKAnisotropicC2(xy, dmax, bd_poly)
    rk.process()

    u = np.linspace(0, dmax, 100)
    k_obs = rk.compute_k(u)
    k_sim = rk.run_permutation(u, niter=nsim)

    return u, k_obs, k_sim


def simulate_sepp_on_network(net,
                             t_total=500,
                             num_to_prune=400):
    obj = NetworkHomogBgExponentialGaussianTrig(net)
    obj.t_total = t_total
    obj.num_to_prune = num_to_prune
    obj.run()

    sim_data = obj.data
    return sim_data.space


if __name__ == '__main__':
    OUT_SUBDIR = 'anisotropy_simulation_study/manhattan/uniform_background/'
    out_dir = os.path.join(OUT_DIR, OUT_SUBDIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    """
    STEP 1
    deposit points on the network
    vary the domain shape and/or network bias towards horizontal/vertical roads
    measure anisotropic K and save
    """

    base_domain_length = 5000.
    domain_width_height_ratios = [1.,
                                  2. / 3.,
                                  1. / 2.,
                                  1. / 4.]
    domain_extents = [(0, 0, base_domain_length * t ** 0.5, base_domain_length * t ** 0.5 / t) for t in domain_width_height_ratios]

    row_space = 100.
    col_spaces = [100., 150., 200., 400.]

    vary_domain = {}

    for ext in domain_extents:
        print "Running vary_domain with bounds %s." % str(ext)
        net = simulate.create_grid_network(ext, row_space, col_spaces[0])
        net_pts = simulate_sepp_on_network(net)
        u, k_obs, k_sim = run_anisotropic_k(net_pts)
        vary_domain[ext] = {
            'k_obs': k_obs,
            'k_sim': k_sim,
            'u': u,
        }

    fn = 'simulate_data_vary_domain.dill'
    out_file = os.path.join(out_dir, fn)
    with open(out_file, 'wb') as f:
        dill.dump(vary_domain, f)
    print "Saved."

    del vary_domain
    vary_network = {}

    for cs in col_spaces:
        print "Running vary_network with col space %f." % cs
        net = simulate.create_grid_network(domain_extents[0], row_space, cs)
        net_pts = simulate_sepp_on_network(net)
        u, k_obs, k_sim = run_anisotropic_k(net_pts)
        vary_network[cs] = {
            'k_obs': k_obs,
            'k_sim': k_sim,
            'u': u,
        }

    fn = 'simulate_data_vary_network.dill'
    out_file = os.path.join(out_dir, fn)
    with open(out_file, 'wb') as f:
        dill.dump(vary_network, f)
    print "Saved."