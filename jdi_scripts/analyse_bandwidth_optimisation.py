from jdi.data import consts, boundary
from settings import JDI_OUT_DIR
import os
import pickle
import dill
import numpy as np
import operator
from shapely import ops
from jdi_scripts import optimise_kde_bandwidth_planar as okbp
from jdi_scripts import optimise_kde_bandwidth_network as okbn
from matplotlib import cm


def likelihood_grids_from_raw_data(
        raw_likelihood_arr,
        shape,
        min_ll=np.log(1e-12)):
    """
    raw_likelihood_arr is a list of length N (number of testing days).
    Each element is a np.array (or equivalent) of length M x L
    where M is the number of parameter pairs (=m1 x m2) and L is the number of test data.
    Here, we take logs and enforce a minimum value.
    return: A list of length N with matrices of shape (m1 x m2)
    """
    ll = []

    for x in raw_likelihood_arr:
        # skipped iterations are indicated with a None value
        if x is None:
            continue
        this_ll = np.log(x)
        this_ll[this_ll < min_ll] = min_ll
        # reshape default order should be correct, as 'flat' was used originally
        this_ll = this_ll.sum(axis=1).reshape(shape)
        ll.append(this_ll)
        
    return ll


def load_raw_likelihood_data_one_crime(outdir):
    #out_dir = os.path.join(
    #    OUT_DIR,
    #    'planar_bandwidth_linearexponential',
    #    consts.CRIME_TYPE_NAME_MAP.get(crime_type),
    #)
    files = [fn for fn in os.listdir(outdir) if fn[-5:] == '.dill']
    ll = {}
    for i, fn in enumerate(files):
        ff = os.path.join(outdir, fn)
        k = fn.split('.')[0]
        with open(ff, 'rb') as f:
            tmp = pickle.load(f)
            if i == 0:
                tt = tmp['tt']
                dd = tmp['dd']
            ll[k] = likelihood_grids_from_raw_data(tmp['ll'], tt.shape)
    return tt, dd, ll


# def plot_daily_likelihood_array(tt, dd, ll, plot_shape=(5, 5), cmap=cm.RdBu):
#     from matplotlib import pyplot as plt
#     from plotting import utils
#
#     # base colormap
#     base_vals = np.array([t.min() for t in ll])
#     base_sm = utils.colour_mapper(base_vals, cmap=cmap)
#
#     fig, axs = plt.subplots(*plot_shape, sharex=True, sharey=True, figsize=(12, 8))
#     n = np.prod(plot_shape)
#     for i in range(n):
#         ax = axs.flat[i]
#         z = ll[i]
#         xl, xu = utils.abs_bound_from_rel(z, [0.25, 1.])
#         base_colour = base_sm.to_rgba(z.min())
#         this_cm = utils.custom_colourmap_white_to_colour(base_colour[:3])
#         bins = np.linspace(xl, xu)
#         ax.contourf(tt, dd, z, bins, cmap=this_cm)
#         ii, jj = np.unravel_index(i, plot_shape)
#         if ii == plot_shape[0] - 1:
#             ax.set_xticks([0, tt.max()])
#         if jj == 0:
#             ax.set_yticks([0, dd.max()])
#
#     ax.set_xlim([0, tt.max()])
#     ax.set_ylim([0, dd.max()])
#     plt.tight_layout(pad=0., h_pad=0.02, w_pad=0.5, rect=(0.01, 0.01, 0.99, 0.99))


def plot_daily_likelihood_array(tt, dd, ll, plot_shape=(5, 5), cmap=cm.Reds, show_opt=True):
    from matplotlib import pyplot as plt
    from plotting import utils

    fig, axs = plt.subplots(*plot_shape, sharex=True, sharey=True, figsize=(12, 8))
    n = np.prod(plot_shape)
    for i in range(n):
        ax = axs.flat[i]
        z = ll[i]
        xl, xu = utils.abs_bound_from_rel(z, [0.25, 1.])
        bins = np.linspace(xl, xu)
        ax.contourf(tt, dd, z, bins, cmap=cmap)
        ii, jj = np.unravel_index(i, plot_shape)
        if ii == plot_shape[0] - 1:
            ax.set_xticks([0, tt.max()])
        if jj == 0:
            ax.set_yticks([0, dd.max()])
        if show_opt:
            topt, dopt = compute_optimum_bandwidth(tt, dd, z)
            ax.plot(topt, dopt, 'kx', ms=10, lw=2.5, mew=2)
            # ax.plot([topt, topt], [0, dopt], 'k--')
            # ax.plot([0, topt], [dopt, dopt], 'k--')

    ax.set_xlim([0, tt.max()])
    ax.set_ylim([0, dd.max()])
    plt.tight_layout(pad=0., h_pad=0.02, w_pad=0.5, rect=(0.01, 0.01, 0.99, 0.99))


def compute_and_save_aggregated_likelihood_grids_one_crime(crime_type, subdir):
    indir = os.path.join(
        JDI_OUT_DIR,
        subdir,
        consts.CRIME_TYPE_NAME_MAP.get(crime_type),
        'optimal_bandwidths',        
    )
    tt, dd, ll = load_raw_likelihood_data_one_crime(indir)
    lltot = {}
    for k in ll:
        if len(ll[k]):
            lltot[k] = reduce(operator.add, ll[k])
    outdir = os.path.join(
        JDI_OUT_DIR,
        subdir,
        'aggregated_likelihoods'
    )
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    out_file = os.path.join(
        outdir,
        '%s.pkl' % consts.CRIME_TYPE_NAME_MAP.get(crime_type)
    )
    with open(out_file, 'wb') as f:
        dill.dump((tt, dd, lltot), f)
    return tt, dd, lltot


def compute_optimum_bandwidth(tt, dd, ll):
    """
    Compute the optimum bandwidth pair
    :param ll: single log likelihood array
    """
    assert ll.shape == tt.shape, "Incompatible shapes"
    idx = np.argmax(ll)
    i, j = np.unravel_index(idx, ll.shape)
    return tt[i, j], dd[i, j]
    
    
def plot_aggregated_likelihood_surface(tt, dd, lltot, 
                                       ax=None, 
                                       title=None, 
                                       fmin=0.25,
                                       colorbar=True):
    from matplotlib import pyplot as plt
    from plotting.utils import abs_bound_from_rel
    from analyse_bandwidth_optimisation import compute_optimum_bandwidth

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax = ax or plt.gca()    
    if title is not None:
        ax.set_title(title)
    vmin = abs_bound_from_rel(lltot, fmin)
    vmax = lltot.max()
    bins = np.linspace(vmin, vmax, 50)
    h = ax.contourf(tt, dd, lltot, bins, cmap='Reds')
    topt, dopt = compute_optimum_bandwidth(tt, dd, lltot)
    ax.plot([0, topt], [dopt, dopt], 'k--')
    ax.plot([topt, topt], [0, dopt], 'k--')
    if colorbar:
        plt.colorbar(h, ax=ax)
    return h


def plot_aggregated_likelihood_array(tt, dd, lltot,
                                     boroughs=None,
                                     plot_shape=None,
                                     fmin=0.25,
                                     cmap=cm.Reds):
    if boroughs is None:
        boroughs = ('tx', 'tw', 'fh', 'bs', 'ni', 'ww', 'yr', 'ye', 'ji')
        # boroughs = ('ni', 'qk', 'sx', 'yr', 'ek', 'cw')

    if plot_shape is None:
        plot_shape = (3, 3)

    from matplotlib import pyplot as plt
    from plotting.utils import abs_bound_from_rel

    fig, axs = plt.subplots(*plot_shape, sharex=True, sharey=True, figsize=(12.5, 10.))

    for i, bo in enumerate(boroughs):
        if i == axs.size:
            break
        if bo not in lltot:
            continue
        ax = axs.flat[i]
        v = lltot[bo]
        vmin = abs_bound_from_rel(v, 0.25)
        vmax = v.max()
        bins = np.linspace(vmin, vmax, 50)
        ax.contourf(tt, dd, v, bins, cmap=cmap)
        ax.set_title(consts.BOROUGH_NAME_MAP[bo.upper()])
        topt, dopt = compute_optimum_bandwidth(tt, dd, v)
        ax.plot([0, topt], [dopt, dopt], 'k--')
        ax.plot([topt, topt], [0, dopt], 'k--')

    plt.tight_layout(pad=0.3, rect=(0.04, 0.04, 0.99, 0.98))

    
def combine_and_save_validation_hit_rate_results(crime_type, subdir, method='grid'):
    """
    Load all .dill files in the given sub directory, combine into a dictionary
    save again in one file and return.
    Method is '' for network, 'grid' or 'intersection' for planar.
    """
    if method == '':
        indir = os.path.join(JDI_OUT_DIR,
                            subdir,
                            consts.CRIME_TYPE_NAME_MAP.get(crime_type, crime_type),
                            'validation_hit_rate')
    else:
        indir = os.path.join(JDI_OUT_DIR,
                            subdir,
                            consts.CRIME_TYPE_NAME_MAP.get(crime_type, crime_type),
                            'validation_%s_hit_rate' % method)
    infiles = os.listdir(indir)
    
    if method == '':
        outdir = os.path.join(JDI_OUT_DIR,
                            subdir,
                            'aggregated_hit_rates')
    else:
        outdir = os.path.join(JDI_OUT_DIR,
                            subdir,
                            'aggregated_hit_rates',
                            method)        
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, "%s.pkl" % consts.CRIME_TYPE_NAME_MAP.get(crime_type, crime_type))
    
    x = {}
    y = {}
    
    for fn in infiles:
        full_fn = os.path.join(indir, fn)
        k = fn.split('.')[0]
        with open(full_fn, 'rb') as f:
            a = dill.load(f)
            x[k] = a['x']
            y[k] = a['y']
            
    with open(outfile, 'wb') as f:
        dill.dump({'coverage': x, 'hit_rate': y}, f)
        
    return x, y
    
    
def load_aggregated_validation_hit_rate_results(crime_type, subdir, method='grid'):
    """ method iterpreted as in combine_and_save_validation_hit_rate_results """
    if method == '':
        indir = os.path.join(JDI_OUT_DIR,
                            subdir,
                            'aggregated_hit_rates')
    else:
        indir = os.path.join(JDI_OUT_DIR,
                            subdir,
                            'aggregated_hit_rates',
                            method)  

    infile = os.path.join(indir, "%s.pkl" % consts.CRIME_TYPE_NAME_MAP.get(crime_type, crime_type))
    
    with open(infile, 'rb') as f:
        return dill.load(f)
        
        
def plot_mean_hit_rate_comparison_one_method(coverage_dict, 
                                             hit_rate_dict, 
                                             ax=None,
                                             xmax=0.25):
    """
    Inputs as retrieved by load_aggregated_validation_hit_rate_results
    """
    from matplotlib import pyplot as plt
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    for k in sorted(coverage_dict.keys()):
        ax.plot(coverage_dict[k].mean(axis=0), hit_rate_dict[k].mean(axis=0), 
        label=consts.BOROUGH_NAME_MAP[k.upper()])
    ax.set_xlim([0, xmax])
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Hit rate')
    ax.legend(loc='upper left')

    
def plot_optimal_bandwidth_maps(crime_type):
    from matplotlib import pyplot as plt
    from plotting.spatial import plot_shapely_geos
    from plotting.utils import colour_mapper
    
    ttp, ddp, llp = okbp.load_aggregated_results(crime_type)
    ttn, ddn, lln = okbn.load_aggregated_results(crime_type)
    sc_map = colour_mapper(np.unique(ttn), fmin=0.25, fmax=0.9)
    bdy = boundary.get_borough_boundary()
    lon = ops.cascaded_union(bdy.values())
    xmin, ymin, xmax, ymax = lon.bounds
    topt_n = {}; dopt_n = {}; topt_p = {}; dopt_p = {}
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    for k in bdy:
        if (k in llp) and (k in lln):
            topt_p[k], dopt_p[k] = compute_optimum_bandwidth(ttp, ddp, llp[k])
            plot_shapely_geos(bdy[k], ec='k', fc=sc_map.to_rgba(topt_p[k]), ax=axs[0])                        
            topt_n[k], dopt_n[k] = compute_optimum_bandwidth(ttn, ddn, lln[k])
            plot_shapely_geos(bdy[k], ec='k', fc=sc_map.to_rgba(topt_n[k]), ax=axs[1])
            circ_p = bdy[k].centroid.buffer(dopt_p[k], 64)
            plot_shapely_geos(circ_p, ax=axs[0], ec='none', fc='b')
            circ_n = bdy[k].centroid.buffer(dopt_n[k], 64)
            plot_shapely_geos(circ_n, ax=axs[1], ec='none', fc='b')
            
            
        else:
            plot_shapely_geos(bdy[k], ec='k', fc='none', ax=axs[0])                        
            plot_shapely_geos(bdy[k], ec='k', fc='none', ax=axs[1])                        
            
    axs[0].set_xlim([xmin, xmax])
    axs[0].set_ylim([ymin, ymax])    
    axs[0].set_aspect('equal')    
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_frame_on(False)
    axs[1].set_frame_on(False)
    plt.tight_layout(0.)