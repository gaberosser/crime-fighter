import numpy as np
from network import simulate, itn
from networkx import MultiGraph
from shapely.geometry import LineString
from stats import ripley
from analysis.spatial import shapely_rectangle_from_vertices
import os
import dill
from scripts import OUT_DIR

def create_grid_network(domain_extents,
                        row_spacing,
                        col_spacing):
    """
    Create a Manhattan network with horizontal / vertical edges.
    :param domain_extents: (xmin, ymin, xmax, ymax) of the domain
    :param row_spacing: Distance between horizontal edges
    :param col_spacing: Distance between vertical edges
    :return: Streetnet object
    """
    xmin, ymin, xmax, ymax = domain_extents
    # compute edge coords
    y = np.arange(ymin + row_spacing / 2., ymax, row_spacing)
    x = np.arange(xmin + col_spacing / 2., xmax, col_spacing)
    g = MultiGraph()
    letters = []
    aint = ord('a')
    for i in range(26):
        for j in range(26):
            for k in range(26):
                letters.append(chr(aint + i) + chr(aint + j) + chr(aint + k))

    def add_edge(x0, y0, x1, y1):
        if x0 < 0 or y0 < 0 or x1 >= len(x) or y1 >= len(y):
            # no link needed
            return
        k0 = x0 * x.size + y0
        k1 = x1 * x.size + y1
        idx_x0 = letters[k0]
        idx_x1 = letters[k1]
        label0 = idx_x0 + str(y0)
        label1 = idx_x1 + str(y1)
        ls = LineString([
            (x[x0], y[y0]),
            (x[x1], y[y1]),
        ])
        atts = {
            'fid': "%s-%s" % (label0, label1),
            'linestring': ls,
            'length': ls.length,
            'orientation_neg': label0,
            'orientation_pos': label1
        }
        g.add_edge(label0, label1, key=atts['fid'], attr_dict=atts)
        g.node[label0]['loc'] = (x[x0], y[y0])
        g.node[label1]['loc'] = (x[x1], y[y1])

    for i in range(x.size):
        for j in range(y.size):
            add_edge(i, j-1, i, j)
            add_edge(i-1, j, i, j)
            add_edge(i, j, i, j+1)
            add_edge(i, j, i+1, j)

    return itn.ITNStreetNet.from_multigraph(g)


def run_anisotropic_k(net,
                      npt=1000,
                      dmax=400,
                      nsim=100):
    bds = net.extent
    xr = bds[2] - bds[0]
    yr = bds[3] - bds[1]
    buff = 0.01
    bd_poly = shapely_rectangle_from_vertices(
        bds[0] - xr * buff,
        bds[1] - yr * buff,
        bds[2] + xr * buff,
        bds[3] + yr * buff,
    )
    net_pts = simulate.uniform_random_points_on_net(net, npt)
    xy = net_pts.to_cartesian()
    rk = ripley.RipleyKAnisotropic(xy, dmax, bd_poly)
    rk.process()

    # distance and angle vectors
    u = np.linspace(0, dmax, 200)
    phi = [((2 * i + 1) * np.pi / 8, np.pi / 4.) for i in range(8)]

    k_obs = rk.compute_k(u, phi=phi)
    k_sim = rk.run_permutation(u, phi=phi, niter=nsim)

    return u, k_obs, k_sim


if __name__ == '__main__':
    OUT_SUBDIR = 'anisotropy_simulation_study/manhattan/uniform_background/'
    out_dir = os.path.join(OUT_DIR, OUT_SUBDIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

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
        net = create_grid_network(ext, row_space, col_spaces[0])
        u, k_obs, k_sim = run_anisotropic_k(net)
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
        net = create_grid_network(domain_extents[0], row_space, cs)
        u, k_obs, k_sim = run_anisotropic_k(net)
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