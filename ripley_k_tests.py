__author__ = 'gabriel'
from django.db import connection
from analysis import chicago, spatial
import datetime
import numpy as np
from time import time
import dill
from shapely import geometry
import multiprocessing as mp

from database.models import Chicago
from data.models import CartesianSpaceTimeData, CartesianData
from kde import models as kde_models
from point_process.utils import pairwise_differences_indices

SRID = 2028


class RipleyK(object):

    table_name = Chicago._meta.db_table
    kde_class = kde_models.VariableBandwidthNnKdeSeparable

    def __init__(self,
                 max_d,
                 start_date,
                 end_date,
                 domain,
                 srid=SRID,
                 crime_type='burglary'):

        self.data = None
        self.n = None
        self.start_date = start_date
        self.end_date = end_date
        self.max_d = max_d
        self.domain = domain
        self.crime_type = crime_type
        self.srid = srid
        self.T = (end_date - start_date).total_seconds() / float(60 * 60 * 24)
        self.S = self.domain.area
        self.ii = self.jj = self.dt = self.dd = self.edge_corr = None
        self.intensity = None
        self.initial_populate()
        self.kde = None
        # self.set_kde()

    def initial_populate(self):
        cursor = connection.cursor()

        sql = """
        WITH poly AS (
        SELECT ST_GeomFromText('{0}', {1}) as poly
        ),
        a AS (
        SELECT datetime, location, number, ST_X(location) AS x, ST_Y(location) AS y,
        row_number() OVER (ORDER BY datetime) AS id
        FROM {2} a
        WHERE
        (SELECT poly FROM poly) && a.location
        AND
        ST_Contains(
        (SELECT poly FROM poly),
        a.location
        )
        AND LOWER(a.primary_type) = 'burglary'
        """.format(self.domain.wkt, self.srid, self.table_name)

        if self.start_date:
            sql += """AND datetime >= '{0}' """.format(self.start_date.strftime('%Y-%m-%d %H:%M:%S'))
        if self.end_date:
            sql += """AND datetime <= '{0}' """.format(self.end_date.strftime('%Y-%m-%d %H:%M:%S'))

        sql += """
        ORDER BY datetime
        )
        SELECT
        i.id AS id_i,
        j.id AS id_j,
        EXTRACT(DAY FROM (j.datetime - i.datetime)) AS dt,
        ST_Distance(j.location, i.location) as dd,
        CASE WHEN
        ST_DWithin(
        (SELECT ST_Boundary(poly) FROM poly),
        i.location,
        ST_Distance(j.location, i.location)
        )
        THEN
        DIVIDE(
            ST_Length(
                ST_Intersection(
                    (SELECT poly FROM poly), ST_Boundary(ST_Buffer(i.location, ST_Distance(j.location, i.location), 32))
                )
            ),
            (2 * pi() * ST_Distance(j.location, i.location))
        )
        ELSE 1 END AS edge_corr_coeff,
        i.datetime AS ti, ST_X(i.location) as xi, ST_Y(i.location) as yi,
        j.datetime AS tj, ST_X(j.location) as xj, ST_Y(j.location) as yj
        FROM a i
        JOIN a j ON i.datetime <= j.datetime
        AND i.id < j.id
        AND ST_Distance(j.location, i.location) < {0}
        -- AND EXTRACT(DAY FROM (j.datetime - i.datetime)) < {1}
        ORDER BY ti, tj
        """.format(
            self.max_d,
            'this is where max_t would have gone'
        )

        tic = time()
        cursor.execute(sql)
        print "Fetched DB results in %f s" % (time() - tic)
        res = cursor.fetchall()
        self.ii = np.array([x[0] for x in res])
        self.jj = np.array([x[1] for x in res])
        assert np.all(self.jj > self.ii), "Expect j > i for all pairs."
        self.dt = np.array([x[2] for x in res])
        assert np.all(self.dt >= 0), "Found negative time differences"
        self.dd = np.array([x[3] for x in res])
        assert np.all(self.dd >= 0), "Found negative distances"
        self.edge_corr = np.array([x[4] for x in res])
        ti = [x[5] for x in res]
        tj = [x[8] for x in res]
        t0 = min(min(ti), min(tj))
        ti = np.array([[(x - t0).total_seconds() / float(60 * 60 * 24)] for x in ti])
        tj = np.array([[(x - t0).total_seconds() / float(60 * 60 * 24)] for x in tj])
        self.i_data = np.array([(x[6], x[7]) for x in res])
        self.j_data = np.array([(x[9], x[10]) for x in res])
        # self.i_data = np.hstack((ti, xyi))
        # self.j_data = np.hstack((tj, xyj))

        from analysis import chicago
        self.data, _, _ = chicago.get_crimes_by_type(self.crime_type,
                                                     start_date=self.start_date,
                                                     end_date=self.end_date,
                                                     domain=self.domain)
        self.n = len(self.data)

        # homogeneous Poisson estimate of intensity
        self.intensity = self.n / float(self.S)

    def compute_k(self, t):
        ind = (self.dd <= t)
        ii = self.ii[ind]
        jj = self.jj[ind]
        w = 1 / self.edge_corr[ind]
        return  sum(w) / float(self.n) / self.intensity
        # int_i = np.array([self.intensity[t] for t in ii])
        # int_j = np.array([self.intensity[t] for t in jj])
        # nv = sum(self.data[:, 0] <= (self.T - v))
        # return self.n / (self.S * self.T * nv) * sum(1 / (w * int_i * int_j))

    def compute_l(self, t):
        """
        Compute the difference between K and the CSR model
        :param u: Distance threshold
        :param v: Time threshold
        :return:
        """
        k = self.compute_k(t)
        csr = np.pi * t ** 2
        return k - csr

    def run_permutation(self, u, niter=20):
        res = []
        def callback(x):
            res.append(x)
        if np.any(u > self.max_d):
            raise AttributeError('No values of u may be > max_d')
        pool = mp.Pool()
        jobs = []
        for i in range(niter):
            jobs.append(
                pool.apply_async(run_permutation,
                                 args=(u, self.n, self.domain, self.max_d),
                                 callback=callback)
            )
        pool.close()
        pool.join()
        return np.array(res)


def run_permutation(u, n, domain, max_d, n_quad=32):

    np.random.seed()

    i1, j1 = pairwise_differences_indices(n)
    ii = np.concatenate((i1, j1))
    jj = np.concatenate((j1, i1))

    xy = CartesianData.from_args(*spatial.random_points_within_poly(domain, n))
    dd = xy.getrows(ii).distance(xy.getrows(jj)).toarray()
    include_idx = np.where(dd <= max_d)[0]
    d_to_ext = np.array([geometry.Point(xy[ii[i]]).distance(domain.exterior) for i in include_idx])
    edge_corr_factor = np.ones(dd.size)
    for i in range(d_to_ext.size):
        # no need to do anything if the circle is entirely within the domain
        if d_to_ext[i] > max_d:
            continue
        # if it is, compute the edge correction factor
        idx = include_idx[i]
        poly = geometry.Point(xy[ii[idx]]).buffer(dd[idx], n_quad)
        edge_corr_factor[idx] = poly.exterior.intersection(domain).length / (2 * np.pi * dd[idx])

    k_func = lambda t: domain.area * (1 / edge_corr_factor[dd <= t]).sum() / n ** 2
    l_func = lambda t: k_func(t) - np.pi * t ** 2

    return np.array([l_func(t) for t in u])

if __name__ == '__main__':

    max_d = 500
    start_date = datetime.date(2011, 3, 1)
    end_date = start_date + datetime.timedelta(days=366)
    domains = chicago.get_chicago_side_polys(as_shapely=True)

    # define a vector of threshold distances
    u = np.linspace(0, max_d, 100)

    domain_mapping = {
        'chicago_south': 'South',
        'chicago_southwest': 'Southwest',
        'chicago_west': 'West',
        'chicago_northwest': 'Northwest',
        'chicago_north': 'North',
        'chicago_central': 'Central',
        'chicago_far_north': 'Far North',
        'chicago_far_southwest': 'Far Southwest',
        'chicago_far_southeast': 'Far Southeast',
    }

    REGIONS = (
        'chicago_central',
        'chicago_far_southwest',
        'chicago_northwest',
        'chicago_southwest',
        'chicago_far_southeast',
        'chicago_north',
        'chicago_south',
        'chicago_west',
        'chicago_far_north',
    )

    CRIME_TYPES = (
        'burglary',
        'assault',
    )

    for r in REGIONS:
        for ct in CRIME_TYPES:
            domain = domains[domain_mapping[r]].simplify(5)
            obj = RipleyK(max_d, start_date, end_date, domain, crime_type=ct)

            l_obs = l_sim = None
            l_obs = np.array([obj.compute_l(t) for t in u])
            l_sim = obj.run_permutation(u, niter=500)

            with open('ripley_%s_%s.pickle' % (r, ct), 'w') as f:
                dill.dump(
                    {'obj': obj,
                     'l_obs': l_obs,
                     'l_sim': l_sim},
                    f
                )
            print "Completed %s %s" % (r, ct)


