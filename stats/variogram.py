__author__ = 'gabriel'
from django.contrib.gis import geos
import numpy as np
from analysis import cad
from point_process.utils import linkages, linkage_func_separable
from point_process import simulate
from data.models import CartesianSpaceTimeData, DataArray
from kde import models as kde_models
from matplotlib import pyplot as plt

b_sim = True

def ripley_correction(centre, radius, poly, n_pt=12):
    ## NB assuming that the point is INSIDE the domain to begin with
    pt = geos.Point(centre)
    if pt.distance(poly.boundary) >= radius:
        return 1.
    d = pt.buffer(radius, quadsegs=n_pt).boundary.intersection(poly).length
    return d / (2 * np.pi * radius)


def simulation():
    decay_const =  1.
    intensity = 1000.
    x_decay_mean = 0.5
    inv_func_t = lambda x: -1.0 / decay_const * np.log((intensity - x)/intensity)
    sim_times = simulate.nonstationary_poisson(inv_func=inv_func_t, t_max=1)
    x = np.random.exponential(scale=x_decay_mean, size=sim_times.size)
    y = np.random.random(sim_times.size)
    out = x > 1.
    while np.any(out):
        xn = np.random.exponential(scale=2, size=out.sum())
        x[out] = xn
        out = x > 1.
    return CartesianSpaceTimeData.from_args(sim_times, x, y)

if b_sim:
    max_t = 0.1
    max_d = 0.1
    du = 0.01
    dv = 0.01
    number_nn = [25, 25]
    poly = geos.Polygon([
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),
        (0, 0)
    ])
    res_all = simulation()
    cid_all = np.arange(res_all.ndata)

else:
    max_t = 90
    max_d = 500
    du = 50
    dv = 5
    number_nn = [15, 100]
    simplification_tol = 20  # for speedup in intersection etc.

    full_poly = cad.get_camden_region()
    poly = full_poly.simplify(simplification_tol)
    bdy = poly.boundary

    res_all, t0, cid_all = cad.get_crimes_by_type(nicl_type=3)
    res_all = CartesianSpaceTimeData(res_all)
    cid_all = np.array(sorted(cid_all))



S = poly.area
T = np.ptp(res_all.time)
n = res_all.ndata

# estimate intensity with KDE
scott_spatial_bandwidth = res_all.space.data.std(axis=0, ddof=1) * res_all.ndata ** (-1. / float(2 + 4))
my_temporal_bandwidth = 0.1
k = kde_models.FixedBandwidthKdeSeparable(res_all, bandwidths=list(scott_spatial_bandwidth) + [my_temporal_bandwidth])

# k = kde_models.VariableBandwidthNnKdeSeparable(res_all, number_nn=number_nn)

t1 = max(res_all.time)

linkage_fun = linkage_func_separable(max_t, max_d)
idx_source, idx_target = linkages(res_all, linkage_fun)

# remove self-matches
idx = cid_all[idx_target] != cid_all[idx_source]
idx_source = idx_source[idx]
idx_target = idx_target[idx]

dt = res_all.time.getrows(idx_target) - res_all.time.getrows(idx_source)
dd = res_all.space.getrows(idx_target).distance(res_all.space.getrows(idx_source))

# set up scales of interest
us = np.arange(du, du + max_d, du)
vs = np.arange(dv, dv + max_t, dv)

uu, vv = np.meshgrid(us, vs, copy=False)
kst = np.zeros_like(uu)
omegas = []
nvs = []
number_links = []
l_sources = []
l_targets = []

for i in range(vs.size):
    for j in range(us.size):
        u = uu[i, j]
        v = vv[i, j]

        nv = float(np.sum(res_all.time <= (t1 - v)))
        nvs.append(nv)
        this_idx = (dt.toarray(0) <= v) & (dd.toarray(0) <= u)
        number_links.append(this_idx.sum())

        this_idx_source = idx_source[this_idx]
        this_idx_target = idx_target[this_idx]
        centres = res_all.space.getrows(this_idx_source)
        radii = dd[this_idx]
        omega = np.array([ripley_correction(list(c), r, poly) for (c, r) in zip(centres, radii.flat)])

        omegas.append(omega)

        l_source = k.pdf(res_all.getrows(this_idx_source), normed=False)
        l_target = k.pdf(res_all.getrows(this_idx_target), normed=False)

        l_sources.append(l_source)
        l_targets.append(l_target)

        kst[i, j] = n / (nv * S * T) * np.sum(1 / (omega * l_source * l_target))

# plots
plt.figure()
plt.contourf(uu, vv, kst - np.pi * uu**2 * vv, 50)
plt.colorbar()

xy = DataArray.from_meshgrid(*np.meshgrid(np.linspace(res_all.toarray(1).min(), res_all.toarray(1).max(), 50),
                                          np.linspace(res_all.toarray(2).min(), res_all.toarray(2).max(), 50)))
zz = k.partial_marginal_pdf(xy, normed=False)
plt.figure()
plt.contourf(xy.toarray(0), xy.toarray(1), zz, 50)
plt.colorbar()

t = np.linspace(res_all.toarray(0).min(), res_all.toarray(0).max(), 500)
plt.figure()
plt.plot(t, k.marginal_pdf(t, dim=0, normed=False))

# simulation ONLY
if b_sim:
    true_int = lambda xyz: 250 / ((1 - np.exp(-1)) * (1 - np.exp(-2))) * np.exp(-2 * xyz.toarray(1) - xyz.toarray(0))
    kst_sim = np.zeros_like(uu)
    for i in range(vs.size):
        for j in range(us.size):
            u = uu[i, j]
            v = vv[i, j]

            nv = float(np.sum(res_all.time <= (t1 - v)))
            this_idx = (dt.toarray(0) <= v) & (dd.toarray(0) <= u)

            this_idx_source = idx_source[this_idx]
            this_idx_target = idx_target[this_idx]
            centres = res_all.space.getrows(this_idx_source)
            radii = dd[this_idx]
            omega = np.array([ripley_correction(list(c), r, poly) for (c, r) in zip(centres, radii.flat)])

            l_source = true_int(res_all.getrows(this_idx_source))
            l_target = true_int(res_all.getrows(this_idx_target))

            kst_sim[i, j] = n / (nv * S * T) * np.sum(1 / (omega * l_source * l_target))


    plt.figure()
    plt.contourf(uu, vv, kst_sim - np.pi * uu**2 * vv, 50)
    plt.colorbar()

    zz_sim = 250 / ((1 - np.exp(-2))) * np.exp(-2 * xy.toarray(0))
    plt.figure()
    plt.contourf(xy.toarray(0), xy.toarray(1), zz_sim, 50)
    plt.colorbar()