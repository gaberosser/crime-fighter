__author__ = 'gabriel'
from django.db import connection, transaction
from django.contrib.gis.db import models
from django.contrib.contenttypes.management import update_contenttypes
from django.core.management import call_command
from database.models import Cad
from database.logic import clean_dedupe_cad
import numpy as np
import pp

class CadCircles(models.Model):
    cad_i = models.ForeignKey(Cad, to_field='id', help_text='Linked central CAD entry', null=False, blank=False,
                              related_name='cad_i_set')
    cad_j = models.ForeignKey(Cad, to_field='id', help_text='Linked relative CAD entry', null=False, blank=False,
                              related_name='cad_j_set')
    radius = models.FloatField(verbose_name='circle radius')
    linear_ring = models.LineStringField(srid=27700)

    objects = models.GeoManager()

    class Meta:
        unique_together = ('cad_i_id', 'cad_j_id') # no duplicate pairings


def _create_cad_circle(pqset0, pqset1):
    import numpy as np
    from analysis.models import CadCircles
    from django.contrib.gis.geos import LineString
    # create cad circles for all pairs of cad events provided
    map_d = np.array([(a.att_map.coords[0] - b.att_map.coords[0], a.att_map.coords[1] - b.att_map.coords[1]) for a, b in zip(pqset0, pqset1)], dtype=float)
    radii = np.sqrt(np.sum(map_d ** 2, axis=1))
    # import ipdb; ipdb.set_trace()

    res = []
    for i in range(radii.size):
        if radii[i] < 1e-12:
            continue
        lr = pqset0[i].att_map.buffer(radii[i]).exterior_ring
        d = {
            'cad_i': pqset0[i],
            'cad_j': pqset1[i],
            'radius': radii[i],
            'linear_ring': LineString(lr.coords),
            }
        res.append(CadCircles(**d))

    try:
        CadCircles.objects.bulk_create(res)
    except Exception as exc:
        print repr(exc)
        raise exc
    return [(x.cad_i.id, x.cad_j.id) for x in res]


def populate_cad_circles(nicl_type=None):
    CadCircles.objects.all().delete()
    qset = clean_dedupe_cad(nicl_type=nicl_type)
    x, y = np.meshgrid(qset, qset, copy=False)
    res = []
    count = 0
    job_server = pp.Server()
    chunksize = 200000
    xf = x.flatten()
    yf = y.flatten()
    total = xf.size

    jobs = []

    for i in range(int(total / chunksize) + 1):
        # _create_cad_circle(xf[(i * chunksize):((i+1) * chunksize)], yf[(i * chunksize):((i+1) * chunksize)])
        j = job_server.submit(_create_cad_circle, (xf[(i * chunksize):((i+1) * chunksize)], yf[(i * chunksize):((i+1) * chunksize)]), (), ())
        jobs.append(j)
        print "Dispatched chunk %d -> %d" % (i*chunksize, (i+1)*chunksize)

    res = []
    for i, j in enumerate(jobs):
        res.append(j())
        print "Completed job %d / %d" % (i, len(jobs))

    return res
