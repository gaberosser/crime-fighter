from network import plots
from plotting.spatial import plot_shapely_geos, plot_shaded_regions
from database.birmingham.loader import load_network, BirminghamCrimeFileLoader, load_boundary_file
import dill
import pickle
import numpy as np
from validation import validation, hotspot
import datetime
from scripts import OUT_DIR
import os
from matplotlib import pyplot as plt
from analysis.spatial import shapely_rectangle_from_vertices

# will need to recompute the grid unfortunately
START_DATE = datetime.date(2013, 7, 1)
START_DAY_NUMBER = 240
GRID_LENGTH = 150
N_SAMPLES_PER_GRID = 15

SIZE_M = 5000
XMIN = 403729
YMIN = 286571
XMAX = XMIN + SIZE_M
YMAX = YMIN + SIZE_M
ZOOM_BUFF = 1250

# ZOOM_EXTENT = (
#     403729,
#     285571,
#     408694,
#     288913
# )


def compute_and_save_planar_grid(outfile='planar_grid_prediction_sample_units.dill'):
    # have run this now, but for the record it's how we can recompute the planar grid
    poly = load_boundary_file()
    vb_planar = validation.ValidationIntegration(np.random.rand(10, 3), None, spatial_domain=poly, include_predictions=True)
    vb_planar.set_sample_units(GRID_LENGTH, N_SAMPLES_PER_GRID)
    sample_unit_polys = vb_planar.roc.grid_polys
    sample_unit_extent = vb_planar.roc.sample_units
    with open(os.path.join(OUT_DIR, 'birmingham', outfile), 'w') as f:
        dill.dump({'extent': sample_unit_extent, 'polys': sample_unit_polys}, f)


def load_prediction_values_and_sample_units():
    files = (
        'planar_prediction_sample_units.dill',
        'planar_prediction_values.dill',
        'network_prediction_sample_units.dill',
        'network_prediction_values.dill'
    )
    res = []
    for fn in files:
        with open(os.path.join(OUT_DIR, 'birmingham', fn), 'r') as f:
            res.append(pickle.load(f))

    return res


if __name__ == '__main__':
    # load existing results: net sample units and prediction values, grid sample units and prediction values
    # grid sample units have been recomputed using compute_planar_grid
    grid_units, grid_vals, net_units, net_vals = load_prediction_values_and_sample_units()

    # reduce the network to enable faster plotting
    bbox = shapely_rectangle_from_vertices(XMIN, YMIN, XMAX, YMAX)
    net = net_units[0].graph
    net_reduced = net.within_boundary(bbox)

    # find remaining sample units
    fids = set([e.fid for e in net_reduced.edges()])
    net_idx = [i for i, e in enumerate(net_units) if e.fid in fids]
    net_units_reduced = [net_units[i] for i in net_idx]
    net_vals_reduced = net_vals[:, net_idx]

    # repeat for grid results
    grid_idx = [i for i, p in enumerate(grid_units['polys']) if p.intersects(bbox)]
    grid_polys_reduced = [grid_units['polys'][i] for i in grid_idx]
    grid_vals_reduced = grid_vals[:, grid_idx]

    grid_polys_reduced = [i for p in grid_units['polys'] if p.intersects(bbox)]
    grid_lines_x = set()
    grid_lines_y = set()
    for t in grid_units['extent']:
        if (t[2] >= XMIN) & (t[3] >= YMIN) & (t[0] <= XMAX) & (t[1] <= YMAX):
            grid_lines_x.add(t[0])
            grid_lines_x.add(t[2])
            grid_lines_y.add(t[1])
            grid_lines_y.add(t[3])
    grid_x = sorted(list(grid_lines_x))
    grid_y = sorted(list(grid_lines_y))
    grid_xmin = min(grid_lines_x)
    grid_xmax = max(grid_lines_x)
    grid_ymin = min(grid_lines_y)
    grid_ymax = max(grid_lines_y)

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 10))

    day_idx = (0, 60)

    for i in range(2):

        # net plot
        ax = axs[0, i]
        j = day_idx[i]

        plots.plot_network_density([t.linestring for t in net_units_reduced],
                                   net_vals_reduced[j],
                                   colorbar=False,
                                   cmap='Reds',
                                   ax=ax)
        for gy in grid_y:
            ax.plot([grid_xmin, grid_xmax], [gy, gy], 'k-', lw=2, alpha=0.25)
        for gx in grid_x:
            ax.plot([gx, gx], [grid_ymin, grid_ymax], 'k-', lw=2, alpha=0.25)

        # get highlighted grid squares
        pick_idx = np.argsort(grid_vals_reduced[j])[::-1][:50]
        pick_polys = [grid_polys_reduced[k] for k in pick_idx]

        _ = plot_shapely_geos(pick_polys, ax=ax, edgecolor='b', lw=2, facecolor='none')

        # grid plot
        ax = axs[1, i]
        plots.plot_network_edge_lines([t.linestring for t in net_units_reduced],
                                      ax=ax,
                                      alpha=1.,
                                      colorbar=False)
        plot_shaded_regions(grid_polys_reduced, grid_vals_reduced[j],
                            ax=ax,
                            cmap=plt.cm.Reds,
                            fmax=0.99,
                            colorbar=False,
                            scale_bar=None,
                            alpha=0.5
                            )

        _ = plot_shapely_geos(pick_polys, ax=ax, edgecolor='b', lw=2, facecolor='none')

        # reset extent to cut off frayed edges
        ax.set_xlim(XMIN + ZOOM_BUFF, XMAX - ZOOM_BUFF)
        ax.set_ylim(YMIN + ZOOM_BUFF, YMAX - ZOOM_BUFF)