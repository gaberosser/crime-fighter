__author__ = 'gabriel'
from django.db import connection
from analysis import chicago
import datetime
import numpy as np
from time import time

from database.models import Chicago
from data.models import CartesianSpaceTimeData
from kde import models as kde_models

SRID = 2028

start_date = datetime.date(2011, 3, 1)
end_date = start_date + datetime.timedelta(days=366)
domain = chicago.get_chicago_side_polys(as_shapely=True)['South'].simplify(1)
crime_type = 'burglary'

max_d = 500
max_t = 90




class Stik(object):

    table_name = Chicago._meta.db_table
    kde_class = kde_models.VariableBandwidthNnKdeSeparable

    def __init__(self,
                 max_t,
                 max_d,
                 start_date,
                 end_date,
                 domain,
                 srid=2028,
                 crime_type='burglary'):

        self.data = None
        self.start_date = start_date
        self.end_date = end_date
        self.max_t = max_t
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
        self.set_kde()

    @property
    def n(self):
        return len(self.data)

    @property
    def kde_kwargs(self):
        return {
            'number_nn': [100, 15],
            'strict': False
        }

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
        AND EXTRACT(DAY FROM (j.datetime - i.datetime)) < {1}
        ORDER BY ti, tj
        """.format(
            self.max_d,
            self.max_t
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
        xyi = np.array([(x[6], x[7]) for x in res])
        xyj = np.array([(x[9], x[10]) for x in res])
        self.i_data = np.hstack((ti, xyi))
        self.j_data = np.hstack((tj, xyj))

        from analysis import chicago
        self.data, _, _ = chicago.get_crimes_by_type(self.crime_type,
                                                     start_date=self.start_date,
                                                     end_date=self.end_date,
                                                     domain=self.domain)

    def set_kde(self):
        self.kde = self.kde_class(self.data, **self.kde_kwargs)

    def estimate_intensity(self):
        self.intensity = {}
        lookup_ind = []
        lookup_data = []
        for (i, itxy) in zip(self.ii, self.i_data):
            if i in lookup_ind:
                continue
            lookup_ind.append(i)
            lookup_data.append(itxy)
        for (j, jtxy) in zip(self.jj, self.j_data):
            if j in lookup_ind:
                continue
            lookup_ind.append(j)
            lookup_data.append(jtxy)

        # intensity = self.kde.pdf(CartesianSpaceTimeData(np.array(lookup_data)), normed=False)
        intensity = self.kde.partial_marginal_pdf(CartesianSpaceTimeData(np.array(lookup_data)).space, normed=False) / self.T
        self.intensity = dict(
            [(ix, v) for ix, v in zip(lookup_ind, intensity)]
        )

    def stik(self, u, v):
        ind = (self.dd <= u) & (self.dt <= v)
        ii = self.ii[ind]
        jj = self.jj[ind]
        w = self.edge_corr[ind]
        int_i = np.array([self.intensity[t] for t in ii])
        int_j = np.array([self.intensity[t] for t in jj])
        nv = sum(self.data[:, 0] <= (self.T - v))
        return self.n / (self.S * self.T * nv) * sum(1 / (w * int_i * int_j))

    def stik_rel_csr(self, u, v):
        """
        Compute the difference between STIK and the CSR model
        :param u: Distance threshold
        :param v: Time threshold
        :return:
        """
        k = self.stik(u, v)
        csr = np.pi * u ** 2 * v
        return k - csr