__author__ = 'gabriel'
import datetime
from database import models
from shapely import wkb
import numpy as np
SRID = models.SRID['uk']


def get_boundary(srid=SRID):
    obj = models.ArealUnit()
    if srid is not None:
        fields = ('ST_Transform(mpoly, {0})'.format(srid),)
    else:
        fields = ('mpoly',)
    res = obj.select(
        where_dict={'name': "'Birmingham'", 'type': "'city boundary'"},
        fields=fields,
        convert_to_dict=False)
    return wkb.loads(res[0][0], hex=True)



def get_crimes(crime_type=None,
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
    :param domain: shapely object
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

    obj = models.Birmingham()

    where_dict = {}

    if crime_type:
        where_dict['LOWER(type)'] = "*LIKE '{0}'".format(crime_type.lower())
    if start_date:
        where_dict['datetime_end'] = "*>= '{0}'".format(start_date.strftime('%Y-%m-%d %H:%M:%S'))
    if end_date:
        where_dict['"datetime_start"'] = "*<= '{0}'".format(end_date.strftime('%Y-%m-%d %H:%M:%S'))
    if domain:
        s = "ST_Intersects(location, ST_GeomFromText('{0}', {1}))".format(domain.wkt, SRID)
        where_dict[s] = '*'
    where_dict.update(where_kwargs)
    fields = ('crime_number', 'datetime_start', 'datetime_end', 'ST_X(location)', 'ST_Y(location)')
    res = obj.select(where_dict=where_dict or None, fields=fields)

    if len(res) == 0:
        return

    cid = np.array([x['crime_number'] for x in res])
    n = cid.size

    # there should be no repeats (verified manually), but doesn't hurt to filter anyway
    cid, idx = np.unique(cid, return_index=True)
    res = [res[i] for i in idx]

    t0 = min([x['datetime_start'] for x in res])
    xy = np.array([(res[i]['ST_X(location)'], res[i]['ST_Y(location)']) for i in range(len(res))])
    t = []
    for x in res:
        sd = x['datetime_start']
        ed = x['datetime_end']
        dt = datetime.timedelta(seconds=(ed - sd).total_seconds() * 0.5)
        if convert_dates:
            t.append([((sd + dt) - t0).total_seconds() / float(60 * 60 * 24)])
        else:
            t.append([sd + dt])
    t = np.array(t)
    res = np.hstack((t, xy))

    # sort data
    sort_idx = np.argsort(res[:, 0])
    res = res[sort_idx]
    cid = cid[sort_idx]

    return res, t0, cid