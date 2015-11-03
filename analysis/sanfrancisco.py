__author__ = 'gabriel'
import warnings
import collections
from django.contrib.gis.geos import Polygon, Point
import numpy as np
import datetime
from time import time
from database import models
from point_process import estimation, models as pp_models, validate
from analysis import spatial
from shapely import wkb
import settings
import os
from django.db import connection
from matplotlib import pyplot as plt
import dill
import copy
SRID = 26943


def get_crimes_by_type(crime_type='burglary',
                       start_date=None,
                       end_date=None,
                       domain=None,
                       convert_dates=True,
                       **where_kwargs):
    """
    Get all matching crimes from the Chicago dataset
    :param crime_type:
    :param start_date:
    :param end_date:
    :param domain: geos.GEOSGeometry object
    :param convert_dates: If True, dates are converted to the number of days after t0 (float)
    :param where_strs:
    :return:
    """
    if start_date and not isinstance(start_date, datetime.datetime):
        # start_date refers to 00:00 onwards
        start_date = datetime.datetime.combine(start_date, datetime.time(0))
    if end_date and not isinstance(end_date, datetime.datetime):
        # end_date refers to 23:59:59 backwards
        end_date = end_date + datetime.timedelta(days=1)
        end_date = datetime.datetime.combine(end_date, datetime.time(0)) - datetime.timedelta(seconds=1)

    obj = models.SanFrancisco()

    where_dict = {
        'LOWER(category)': "*LIKE '{0}'".format(crime_type.lower())
    }


    # sql = """ SELECT "id", datetime, ST_X(location), ST_Y(location) FROM database_chicago
    #           WHERE LOWER(category) LIKE '%{0}%' """.format(crime_type.lower())
    if start_date:
        where_dict['datetime'] = "*>= '{0}'".format(start_date.strftime('%Y-%m-%d %H:%M:%S'))
        # sql += """AND datetime >= '{0}' """.format(start_date.strftime('%Y-%m-%d %H:%M:%S'))
    if end_date:
        where_dict['"datetime"'] = "*<= '{0}'".format(end_date.strftime('%Y-%m-%d %H:%M:%S'))
        # sql += """AND datetime <= '{0}' """.format(end_date.strftime('%Y-%m-%d %H:%M:%S'))
    if domain:
        s = "ST_Intersects(location, ST_GeomFromText('{0}', {1}))".format(domain.wkt, SRID)
        where_dict[s] = '*'
        # sql += """AND ST_Intersects(location, ST_GeomFromText('{0}', {1}))"""\
        #     .format(domain.wkt, SRID)
    where_dict.update(where_kwargs)
    # for x in where_strs.values():
    #     sql += """AND {0}""".format(x)
    res = obj.select(where_dict, fields=('incident_number', 'datetime', 'ST_X(location)', 'ST_Y(location)'))

    if len(res) == 0:
        return

    # cursor.execute(sql)
    # res = cursor.fetchall()
    cid = np.array([x['incident_number'] for x in res])

    # identify any repeated incident numbers and choose one (doesn't matter which, space and time components are copied)
    cid, idx = np.unique(cid, return_index=True)
    res = [res[i] for i in idx]


    t0 = min([x['datetime'] for x in res])
    xy = np.array([(res[i]['ST_X(location)'], res[i]['ST_Y(location)']) for i in range(len(res))])
    if convert_dates:
        t = np.array([[(x['datetime'] - t0).total_seconds() / float(60 * 60 * 24)] for x in res])
    else:
        t = np.array([[x['datetime']] for x in res])
    res = np.hstack((t, xy))

    # sort data
    sort_idx = np.argsort(res[:, 0])
    res = res[sort_idx]
    cid = cid[sort_idx]

    return res, t0, cid


def get_boundary():
    obj = models.SanFranciscoDivision()
    wkb_hex_str = obj.select(fields=('mpoly',))[0]['mpoly']
    mpoly = wkb.loads(wkb_hex_str, hex=True)
    return mpoly


def create_grid_squares_shapefile(outfile, side_length=250):
    mpoly = get_boundary()
    ipoly, fpoly, full = spatial.create_spatial_grid(mpoly, side_length)
    field_description = {'id': {'fieldType': 'N'}}
    id = range(1, len(fpoly) + 1)
    full_polys = [spatial.shapely_rectangle_from_vertices(*t) for t in fpoly]
    spatial.write_polygons_to_shapefile(outfile,
                                        full_polys,
                                        field_description=field_description,
                                        id=id)


if __name__ == "__main__":
    start_date = datetime.date(2011, 3, 1)
    end_date = datetime.date(2012, 3, 1)
    data, t0, cid = get_crimes_by_type('burglary', start_date=start_date, end_date=end_date)

    max_t=90
    max_d=500,
    seed=43

    est_fun = lambda x, y: estimation.estimator_bowers_fixed_proportion_bg(x, y, ct=1, cd=10, frac_bg=0.5)
    # trigger_kde_kwargs = {'strict': False}
    trigger_kde_kwargs = {'bandwidths': [10, 40, 40]}
    bg_kde_kwargs = {'strict': False}

    sepp = pp_models.SeppStochasticNnBgFixedTrigger(data=data,
                                                    max_delta_t=max_t,
                                                    max_delta_d=max_d,
                                                    seed=seed,
                                                    estimation_function=est_fun,
                                                    trigger_kde_kwargs=trigger_kde_kwargs,
                                                    bg_kde_kwargs=bg_kde_kwargs)

    # sepp = pp_models.SeppStochasticNn(data=data,
    #                                   max_delta_t=max_t,
    #                                   max_delta_d=max_d,
    #                                   seed=seed,
    #                                   estimation_function=est_fun,
    #                                   trigger_kde_kwargs=trigger_kde_kwargs,
    #                                   bg_kde_kwargs=bg_kde_kwargs)

    ps = sepp.train(niter=50)
