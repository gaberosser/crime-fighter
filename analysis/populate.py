__author__ = 'gabriel'
import multiprocessing
from django.db import IntegrityError, transaction
import numpy as np
import operator
from time import time
from database.logic import clean_dedupe_cad
from database.models import Division
from analysis.models import CadCircles
from django.contrib.gis.geos import LineString, Polygon

CHUNKSIZE = 100000

domain = Division.objects.filter(type='cad_250m_grid').unionagg().exterior_ring

qset = clean_dedupe_cad(nicl_type=3)
qset_ids = np.array([x.id for x in qset], dtype=int)
att_maps = [x.att_map for x in qset]
dist_to_bdy = np.array([x.distance(domain) for x in att_maps])

idx_i, idx_j = np.meshgrid(range(len(qset)), range(len(qset)))

# att_map_x, att_map_y = np.meshgrid(att_maps, att_maps, copy=False)
coords_x = [x.coords[0] for x in att_maps]
coords_y = [x.coords[1] for x in att_maps]
x1, x2 = np.meshgrid(coords_x, coords_x, copy=False)
dx = np.array(x1 - x2, dtype=float)
y1, y2 = np.meshgrid(coords_y, coords_y, copy=False)
dy = np.array(y1 - y2, dtype=float)
radii = np.sqrt(dx**2 + dy**2)

# x, y = np.meshgrid(coords_x, coords_y, copy=False)


def create_circle_coords(centres_x, centres_y, rad, npt=20):
    assert centres_x.size == centres_y.size == rad.size
    th = np.linspace(0, 2 * np.pi, npt + 1)
    res_x = np.tile(np.cos(th), (centres_x.size, 1)).transpose() * rad + centres_x.flatten()
    res_y = np.tile(np.sin(th), (centres_x.size, 1)).transpose() * rad + centres_y.flatten()
    res_x = res_x.transpose()
    res_y = res_y.transpose()
    # manually impose equality on last coords
    res_x[:, -1] = res_x[:, 0]
    res_y[:, -1] = res_y[:, 0]
    return res_x, res_y


def _create_cad_circle(i):
    print i
    sidx = slice(i*CHUNKSIZE, (i+1)*CHUNKSIZE)
    t_idx_i = idx_i.flat[sidx]
    t_idx_j = idx_j.flat[sidx]

    this_radii = radii[t_idx_i, t_idx_j]
    # remove entries where there is no need to compute this quantity
    this_distance_radii = dist_to_bdy[t_idx_i]
    to_keep_idx = this_radii > this_distance_radii
    t_idx_i = t_idx_i[to_keep_idx]
    t_idx_j = t_idx_j[to_keep_idx]
    print "Discarded %d fully embedded circles" % np.sum(~to_keep_idx)

    # print "idx %d despatched" % i*CHUNKSIZE
    # create cad circles for all pairs of cad events provided
    this_radii = radii[t_idx_i, t_idx_j]
    this_ids_i = qset_ids[t_idx_i]
    this_ids_j = qset_ids[t_idx_j]
    this_centres_x = x1[t_idx_i, t_idx_j]
    this_centres_y = y1[t_idx_i, t_idx_j]

    lr_coords_x, lr_coords_y = create_circle_coords(this_centres_x, this_centres_y, this_radii)

    res = []
    for n in range(this_radii.size):
        if this_radii[n] < 1e-12:
            continue
        # lr = att_maps[t_idx_i[n]].buffer(this_radii[n]).exterior_ring
        d = {
            'cad_i_id': this_ids_i[n],
            'cad_j_id': this_ids_j[n],
            'radius': this_radii[n],
            'linear_ring': LineString(zip(lr_coords_x[n, :], lr_coords_y[n, :])),
            }
        # res.append(CadCircles(**d))
        res.append(d)

    return res

    # print "idx %d, commencing bulk_create" % idx.start
    # try:
    #     with transaction.atomic():
    #         CadCircles.objects.bulk_create(res)
    # except Exception as exc:
    #     print repr(exc)
    #     raise exc
    # else:
    #     print "idx %d iteration complete" % idx.start
    # return [(t.cad_i.id, t.cad_j.id) for t in res]


if __name__ == "__main__":
    # res = _create_cad_circle(0)
    niter = int(dx.size / CHUNKSIZE) + 1
    pool = multiprocessing.Pool()
    m = pool.map_async(_create_cad_circle, range(niter))
    pool.close()
    pool.join()
    print "Completed %d chunks, creating objects" % niter
    res = [CadCircles(**t) for t in reduce(operator.add, m.get())]
    print "Commencing bulk create of %d items..." % len(res)
    tic = time()
    with transaction.atomic():
        CadCircles.objects.bulk_create(res)
    print "Completed in %f seconds" % (tic-time())