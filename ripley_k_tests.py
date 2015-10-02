__author__ = 'gabriel'
from django.db import connection
from analysis import chicago, spatial
import datetime
import numpy as np
from time import time
import dill
from shapely import geometry
import multiprocessing as mp
from functools import partial

from data.models import CartesianSpaceTimeData, CartesianData
from kde import models as kde_models
import collections


def edge_correction_wrapper(x, domain=None, n_quad=32):
    return edge_correction(*x, domain=domain, n_quad=n_quad)


def edge_correction(xy, d, domain=None, n_quad=32):
    poly = geometry.Point(xy).buffer(d, n_quad)
    circ = poly.exterior.intersection(domain).length / (2 * np.pi * d)
    area = poly.intersection(domain).area / (np.pi * d ** 2)
    return circ, area


class RipleyK(object):

    kde_class = kde_models.VariableBandwidthNnKdeSeparable
    n_quad = 32

    def __init__(self,
                 data,
                 max_d,
                 domain):

        self.data = CartesianData(data)
        assert self.data.nd == 2, "Input data must be 2D (i.e. purely spatial)"
        self.n = len(data)
        self.max_d = max_d
        self.domain = domain
        self.S = self.domain.area
        self.ii = self.jj = self.dd = self.dphi = None
        self.edge_corr_circ = self.edge_corr_area = None
        self.intensity = self.n / self.S

    def prepare_one(self, i):
        this_i = self.ii[i]
        this_d = self.dd[i]
        poly = geometry.Point(data[this_i]).buffer(this_d, self.n_quad)
        circ = poly.exterior.intersection(self.domain).length / (2 * np.pi * this_d)
        area = poly.intersection(self.domain).area / (np.pi * this_d ** 2)
        return circ, area

    # def compute_edge_correction(self, data=None):
    #     print "Computing edge correction terms..."
    #     data = CartesianData(data) if data is not None else self.data
    #     ii, jj, dd = spatial.spatial_linkages(self.data,
    #                                       self.max_d)
    #     idx_cat = np.argsort(jj)
    #     self.ii = np.concatenate((ii, jj[idx_cat]))
    #     self.jj = np.concatenate((jj, ii[idx_cat]))
    #     self.dd = np.concatenate((dd, dd[idx_cat]))
    #     d_to_ext = np.array([geometry.Point(data[i]).distance(self.domain.exterior) for i in range(self.n)])
    #
    #     self.edge_corr_circ = np.ones(self.dd.size)
    #     self.edge_corr_area = np.ones(self.dd.size)
    #
    #     for i in range(self.ii.size):
    #         this_i = self.ii[i]
    #         this_d = self.dd[i]
    #         # no need to do anything if the circle is entirely within the domain
    #         if d_to_ext[this_i] > this_d:
    #             continue
    #         # if it is, compute the edge correction factor
    #         poly = geometry.Point(data[this_i]).buffer(this_d, self.n_quad)
    #         self.edge_corr_circ[i] = poly.exterior.intersection(self.domain).length / (2 * np.pi * this_d)
    #         self.edge_corr_area[i] = poly.intersection(self.domain).area / (np.pi * this_d ** 2)

    def compute_edge_correction(self, data=None):
        print "Computing edge correction terms..."
        data = CartesianData(data) if data is not None else self.data
        ii, jj, dd = spatial.spatial_linkages(self.data,
                                              self.max_d)
        idx_cat = np.argsort(jj)
        self.ii = np.concatenate((ii, jj[idx_cat]))
        self.jj = np.concatenate((jj, ii[idx_cat]))
        self.dd = np.concatenate((dd, dd[idx_cat]))
        d_to_ext = np.array([geometry.Point(data[i]).distance(self.domain.exterior) for i in range(self.n)])

        self.edge_corr_circ = np.ones(self.dd.size)
        self.edge_corr_area = np.ones(self.dd.size)

        ind = np.where(d_to_ext[self.ii] < self.dd)[0]

        mappable_func = partial(edge_correction_wrapper, n_quad=32, domain=domain)

        pool = mp.Pool()
        res = pool.map_async(mappable_func, ((self.data[self.ii[i]], self.dd[i]) for i in ind)).get(1e100)

        self.edge_corr_circ[ind] = np.array(res)[:, 0]
        self.edge_corr_area[ind] = np.array(res)[:, 1]

    def compute_k(self, u, *args, **kwargs):
        if self.edge_corr_area is None:
            self.compute_edge_correction()
        if not hasattr(u, '__iter__'):
            u = [u]
        res = []
        for t in u:
            ind = (self.dd <= t)
            w = 1 / self.edge_corr_area[ind]  ## TODO: which correction to use here?
            res.append(w.sum() / float(self.n) / self.intensity)
        return np.array(res)

    def compute_l(self, u):
        """
        Compute the difference between K and the CSR model
        :param u: Distance threshold
        :param v: Time threshold
        :return:
        """
        k = self.compute_k(u)
        csr = np.pi * u ** 2
        return k - csr

    def compute_lhat(self, u):
        """
        Lhat is defined as (K / \pi) ^ 0.5
        :param u:
        :return:
        """
        k = self.compute_k(u)
        return np.sqrt(k / np.pi)


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


class RipleyKAnisotropic(RipleyK):

    def __init__(self, *args, **kwargs):
        self.dphi = None
        super(RipleyKAnisotropic, self).__init__(*args, **kwargs)

    def compute_edge_correction(self, data=None):
        super(RipleyKAnisotropic, self).compute_edge_correction(data=data)
        ## If this is slow, can only perform 1/2 the computations by being clever about reversing angles
        self.dphi = data.getrows(self.ii).angle(data.getrows(self.jj))

    def compute_k(self, u, phi, bidirectional=True, *args, **kwargs):
        """
        Compute anisotropic K in which distance is less than u and phi lies in the bins specified.
        :param u: Array of distances.
        :param phi: Array of angle *edges*. The number of angle bins will have length one fewer. MUST BE INCREASING.
        :param bidirectional: If True, each phi range is automatically combined with the equivalent range after adding
        pi. It is up to the user to ensure that ranges do not overlap.
        :return: 2D array, rows represent values in u and cols represent between-values in phi
        """
        assert np.all(np.diff(phi) > 0.), "phi array must be increasing"
        if self.edge_corr_area is None:
            self.compute_edge_correction()
        if not hasattr(u, '__iter__'):
            u = [u]
        # create phi bins
        phi_lower = phi[:-1]
        phi_width = np.diff(phi)
        res = np.zeros((len(u), len(phi_lower)))

        for i in range(len(u)):
            for j in range(len(phi_lower)):
                phi0 = np.mod(phi_lower[j], 2 * np.pi)
                phi1 = phi0 + phi_width[j]
                t = u[i]
                if bidirectional:
                    phi_frac = phi_width[j] / np.pi
                else:
                    phi_frac = phi_width[j] / (2 * np.pi)

                if bidirectional:
                    # compute opposite side angle range
                    dphi = self.dphi.copy
                    rev = (dphi > (np.pi / 2.)) | (dphi < (-np.pi / 2.))
                    dphi[rev] = np.mod(dphi[rev] + np.pi, 2 * np.pi)
                    ind = (self.dd <= t) & (dphi >= phi0) & (dphi < phi1)
                else:
                    ind = (self.dd <= t) & (self.dphi >= phi0) & (self.dphi < phi1)

                # Correct for fraction of circular area / circumference actually covered by the slice
                # This is an APPROXIMATION: in reality, this will vary between slice. However, that makes the
                # calculation of edge correction terms even slower.

                w = 1 / (self.edge_corr_area[ind] * phi_frac)  # using area-based correction
                res[i, j] = w.sum() / float(self.n) / self.intensity

        return res


def run_permutation(u, n, domain, max_d, n_quad=32):

    np.random.seed()

    xy = CartesianData.from_args(*spatial.random_points_within_poly(domain, n))
    ii, jj, dd = spatial.spatial_linkages(xy, max_d)
    idx_cat = np.argsort(jj)
    ii = np.concatenate((ii, jj[idx_cat]))
    jj = np.concatenate((jj, ii[idx_cat]))
    dd = np.concatenate((dd, dd[idx_cat]))

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

    return np.array([k_func(t) for t in u])


if __name__ == '__main__':

    max_d = 500
    geos_simplification = 20  # metres tolerance factor
    # max_d = 5000
    n_sim = 5
    start_date = datetime.date(2011, 3, 1)
    end_date = start_date + datetime.timedelta(days=366)
    domains = chicago.get_chicago_side_polys(as_shapely=True)

    # define a vector of threshold distances
    u = np.linspace(0, max_d, 400)

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
        'chicago_south',
        # 'chicago_central',
        # 'chicago_far_southwest',
        # 'chicago_northwest',
        # 'chicago_southwest',
        # 'chicago_far_southeast',
        # 'chicago_north',
        # 'chicago_west',
        # 'chicago_far_north',
    )

    CRIME_TYPES = (
        'burglary',
        # 'assault',
    )
    res = collections.defaultdict(dict)

    for r in REGIONS:
        for ct in CRIME_TYPES:
            domain = domains[domain_mapping[r]].simplify(geos_simplification)
            data, t0, cid = chicago.get_crimes_by_type(crime_type=ct,
                                                       start_date=start_date,
                                                       end_date=end_date,
                                                       domain=domain)
            tic = time()
            obj = RipleyK(data[:, 1:], max_d, domain)

            k_obs = obj.compute_k(u)
            print "%s, %s, %f seconds" % (domain_mapping[r], ct, time() - tic)
            k_sim = obj.run_permutation(u, niter=n_sim)
            lhat_obs = obj.compute_lhat(u)
            lhat_sim = np.sqrt(k_sim / np.pi)

            res[r][ct] = {'obj': obj,
                          'k_obs': k_obs,
                          'k_sim': k_sim,
                          'lhat_obs': lhat_obs,
                          'lhat_sim': lhat_sim,
                          }

            # with open('ripley_%s_%s.pickle' % (r, ct), 'w') as f:
            #     dill.dump(
            #         {'obj': obj,
            #          'k_obs': k_obs,
            #          'k_sim': k_sim,
            #          'lhat_obs': lhat_obs,
            #          'lhat_sim': lhat_sim,
            #          },
            #         f
            #     )
            print "Completed %s %s" % (r, ct)

    # r = 'chicago_south'
    # ct = 'burglary'
    # domain = domains[domain_mapping[r]].simplify(5)
    # sq_side = 800
    # max_d = np.sqrt(2) * sq_side
    # sq_centre = (448950, 4629000)
    # sq_domain = spatial.shapely_rectangle_from_vertices(sq_centre[0] - sq_side/2,
    #                                                     sq_centre[1] - sq_side/2,
    #                                                     sq_centre[0] + sq_side/2,
    #                                                     sq_centre[1] + sq_side/2)
    # data, t0, cid = chicago.get_crimes_by_type(crime_type=ct,
    #                                            start_date=start_date,
    #                                            end_date=end_date,
    #                                            domain=sq_domain)
    # obj = RipleyK3(data[:, 1:], max_d, sq_domain)
    # u = np.linspace(0, max_d, 100)
    # k_obs = obj.compute_k(u)
    # l_obs = obj.compute_l(u)
    # k_sim = obj.run_permutation(u, niter=n_sim)