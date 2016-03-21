__author__ = 'gabriel'
from analysis import chicago, spatial
from database.chicago import consts as chicago_consts
import datetime
import numpy as np
from time import time
import dill
from shapely import geometry, ops
import multiprocessing as mp
from functools import partial
import operator
from contextlib import closing
from data.models import CartesianSpaceTimeData, CartesianData
from kde import models as kde_models
import collections


def run_dill_encoded(what):
    fun, args = dill.loads(what)
    return fun(*args)


def dill_mapping_function(pool, fun, arg_arr):
    return pool.map_async(run_dill_encoded, (dill.dumps((fun, x)) for x in arg_arr))


def isotropic_edge_correction_wrapper(x, domain=None, n_quad=32, method=None):
    return isotropic_edge_correction(*x, domain=domain, n_quad=n_quad, method=method)


def isotropic_edge_correction(xy, d, domain=None, n_quad=32, method='area'):
    poly = geometry.Point(xy).buffer(d, n_quad)
    if method == "area":
        return poly.intersection(domain).area / (np.pi * d ** 2)
    elif method == "circ":
        return poly.exterior.intersection(domain).length / (2 * np.pi * d)
    else:
        return ValueError("Unrecognised method")


def prepare_data(data, domain, max_d, compute_angles=False):
    """
    Process spatial data to obtain linkages and distance from the boundary for Ripley's K computation.
    :param data:
    :param domain:
    :param max_d:
    :param compute_angles: If True, also calculate dangle, the angle made by the line connecting i to j.
    :return: (parent_idx, child_idx, delta_distance, near_exterior) if compute_angles is False, else
    (parent_idx, child_idx, delta_distance, delta_angle, near_exterior)
    """
    i1, j1, dd = spatial.spatial_linkages(data, max_d)
    idx_cat = np.argsort(j1)
    ii = np.concatenate((i1, j1[idx_cat]))
    jj = np.concatenate((j1, i1[idx_cat]))
    i1 = None
    j1 = None
    dd = np.concatenate((dd, dd[idx_cat]))
    d_to_ext = np.array([geometry.Point(data[i]).distance(domain.exterior) for i in range(len(data))])
    near_exterior = np.where(d_to_ext[ii] < dd)[0]
    if compute_angles:
        dangle = data.getrows(ii).angle(data.getrows(jj)).toarray()
        # mask zero distances to avoid them all appearing at 0 angle
        dangle[dd == 0.] = np.nan
        return ii, jj, dd, dangle, near_exterior
    return ii, jj, dd, near_exterior


class RipleyK(object):

    kde_class = kde_models.VariableBandwidthNnKdeSeparable
    n_quad = 32

    def __init__(self,
                 data,
                 max_d,
                 domain,
                 clip_data=True):
        """

        :param data:
        :param max_d:
        :param domain:
        :param clip_data: If True (default), remove data that lie outside of the domain. This is recommended as they
        will result in a division by zero later on
        :return:
        """
        self.data = CartesianData(data)
        if clip_data:
            to_keep = [i for i, xy in enumerate(self.data.data) if geometry.Point(xy).within(domain)]
            self.data = self.data.getrows(to_keep)
        assert self.data.nd == 2, "Input data must be 2D (i.e. purely spatial)"
        self.n = len(data)
        self.max_d = max_d
        self.domain = domain
        self.S = self.domain.area
        self.ii = self.jj = self.dd = self.dtheta = None
        # self.near_exterior = None
        # self.edge_corr_circ = self.edge_corr_area = None
        self.edge_correction = None
        self.intensity = self.n / self.S

    def process(self):
        # Call after instantiation to prepare all data and compute edge corrections
        self.ii, self.jj, self.dd, self.near_exterior = prepare_data(self.data, self.domain, self.max_d)
        self.edge_correction = self.compute_edge_correction(self.data, self.ii, self.dd, self.domain, self.n_quad)

    @staticmethod
    def compute_edge_correction(data, source_idx, dd, domain, n_quad, *args, **kwargs):
        """
        Compute the edge correction factors for the data centered at loc with spatial extent in dd
        :param data: Full array or DataArray of spatial data
        :param source_idx: Array of source indices
        :param dd: Array of distances, i.e. the circle's radius
        :return:
        """
        data = CartesianData(data, copy=False)

        # compute distance to edge and therefore which points/distances need edge corrections
        d_to_ext = np.array([geometry.Point(data[i]).distance(domain.exterior) for i in range(data.ndata)])
        near_exterior = np.where(d_to_ext[source_idx] < dd)[0]

        # output array of same size as dd
        ec = np.ones_like(dd)

        # can switch the method here
        mappable_func = partial(isotropic_edge_correction_wrapper, n_quad=n_quad, domain=domain, method='area')

        print "Computing edge correction terms..."
        tic = time()
        with closing(mp.Pool()) as pool:
            res = pool.map_async(mappable_func, ((data[source_idx[i]], dd[i]) for i in near_exterior)).get(1e100)
        print "Completed in %f seconds" % (time() - tic)
        ec[near_exterior] = np.array(res)
        return ec

    def compute_k(self, u, dd=None, edge_corr=None, *args, **kwargs):
        if not hasattr(u, '__iter__'):
            u = [u]
        dd = dd if dd is not None else self.dd
        edge_corr = edge_corr if edge_corr is not None else self.edge_correction
        res = []
        for t in u:
            ind = (dd <= t)
            w = 1 / edge_corr[ind]
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
        if np.any(u > self.max_d):
            raise AttributeError('No values of u may be > max_d')
        k = []
        try:
            for i in range(niter):
                data = CartesianData.from_args(*spatial.random_points_within_poly(self.domain, self.n))
                ii, jj, dd, near_exterior = prepare_data(data, self.domain, self.max_d)
                ec = self.compute_edge_correction(data, ii, dd, self.domain, self.n_quad)
                k.append(self.compute_k(u, dd=dd, edge_corr=ec))
        except Exception as exc:
            print repr(exc)
        finally:
            return np.array(k)



class CircleSegment(object):
    def __init__(self, phis, dphis, n_quad=16):
        self.n_quad = n_quad

        if hasattr(phis, '__iter__'):
            self.phis = np.array(phis)
        else:
            self.phis = np.array([phis])

        if hasattr(dphis, '__iter__'):
            self.dphis = np.array(dphis)
            assert self.dphis.size == self.phis.size
        else:
            self.dphis = np.ones_like(self.phis) * dphis

        assert self.dphis.sum() <= 2 * np.pi
        # set filter function
        filt_args = reduce(operator.add, zip(self.phis, self.dphis))
        self.filter = phi_filter_factory(*filt_args)

    @property
    def fraction_circle(self):
        return self.dphis.sum() / (2 * np.pi)

    def area(self, r):
        # sanity check: when sum of dphis is 2 * pi, this should return pi * r ** 2
        return 0.5 * self.dphis.sum() * r ** 2

    def circumference(self, r):
        # sanity check: when sum of dphis is 2 * pi, this should return 2 * pi * r
        return self.dphis.sum() * r

    def polygon(self, loc, r):
        """
        :param loc: (x, y), central coords of circle
        :param r: radius
        :return:
        """
        polys = []
        for p, d in zip(self.phis, self.dphis):
            th = np.linspace(p, p + d, self.n_quad)
            x = np.concatenate((loc[0:1], r * np.cos(th) + loc[0], loc[0:1]))
            y = np.concatenate((loc[1:2], r * np.sin(th) + loc[1], loc[1:2]))
            polys.append(geometry.Polygon(zip(x, y)))
        return ops.cascaded_union(polys)

    def linestring(self, loc, r):
        lines = []
        for p, d in zip(self.phis, self.dphis):
            th = np.linspace(p, p + d, self.n_quad)
            x = r * np.cos(th) + loc[0]
            y = r * np.sin(th) + loc[1]
            lines.append(geometry.LineString(zip(x[1:-1], y[1:-1])))
        return ops.cascaded_union(lines)

    def area_correction(self, loc, r, domain):
        poly = self.polygon(loc, r)
        return poly.intersection(domain).area / self.area(r)

    def circumference_correction(self, loc, r, domain):
        line = self.linestring(loc, r)
        return line.intersection(domain).length / self.circumference(r)

    def contains(self, dtheta):
        """
        Return True if the specified point is within the circular segment(s)
        :param dtheta: Angles of interest (in array if required)
        :return: Boolean, same size as dtheta
        """
        return self.filter(dtheta)


def anisotropic_edge_correction_wrapper(x, domain=None, method=None):
    # x = (loc, radius, segment)
    return anisotropic_edge_correction(*x, domain=domain, method=method)


def anisotropic_edge_correction(xy, d, segment, domain=None, method='area'):
    if method == "area":
        return segment.area_correction(xy, d, domain)
    elif method == "circ":
        return segment.circumference_correction(xy, d, domain)
    else:
        raise ValueError("Unrecognised method")


class RipleyKAnisotropic(RipleyK):

    n_quad = 16

    def __init__(self,
                 data,
                 max_d,
                 domain,
                 segments=None):
        """
        Anisotropic version of Ripley's K
        :param segments: Array of CircleSegment objects, defining angular regions
        :return:
        """
        super(RipleyKAnisotropic, self).__init__(data, max_d, domain)
        self.segments = None
        self.set_segments(segments)

    def set_segments(self, segments=None):
        if segments is None:
            # revert to default behaviour
            dphi = np.pi / 4.
            phi0 = dphi / 2.
            phis = np.arange(phi0, 2 * np.pi, dphi)
            self.segments = [CircleSegment(p, dphi, n_quad=self.n_quad) for p in phis]
        else:
            if not hasattr(segments, '__iter__'):
                segments = [segments]
            self.segments = segments

    @property
    def nsegment(self):
        return len(self.segments)

    def process(self):
        self.ii, self.jj, self.dd, self.dtheta, self.near_exterior = prepare_data(self.data,
                                                                                  self.domain,
                                                                                  self.max_d,
                                                                                  compute_angles=True)
        self.edge_correction = self.compute_edge_correction(self.data,
                                                            self.ii,
                                                            self.dd,
                                                            self.segments,
                                                            self.domain)

    @staticmethod
    def compute_edge_correction(data, source_idx, dd, segments, domain, *args, **kwargs):
        """
        Compute the edge correction factors for the data centered at loc with spatial extent in dd
        :param data: Full array or DataArray of spatial data
        :param source_idx: Array of source indices
        :param dd: Array of distances, i.e. the circle's radius
        :return:
        """
        data = CartesianData(data, copy=False)

        # compute distance to edge and therefore which points/distances need edge corrections
        d_to_ext = np.array([geometry.Point(data[i]).distance(domain.exterior) for i in range(data.ndata)])
        near_exterior = np.where(d_to_ext[source_idx] < dd)[0]

        # output array of same size as dd
        ec = np.ones((dd.size, len(segments)))

        # can switch the method here
        def mappable_func(loc, r, seg):
            return seg.area_correction(loc, r, domain)

        print "Computing edge correction terms..."
        tic = time()
        for j, seg in enumerate(segments):
            iter_args = (
                (data[source_idx[i]], dd[i], domain) for i in near_exterior
            )
            res = np.array([(seg.area_correction(*x)) for x in iter_args])
            ec[near_exterior, j] = np.array(res)
        # with closing(mp.Pool()) as pool:
        #     for j, seg in enumerate(segments):
        #         iter_args = (
        #             (data[source_idx[i]], dd[i], seg) for i in near_exterior
        #         )
        #         ma = dill_mapping_function(
        #             pool,
        #             mappable_func,
        #             ((data[source_idx[i]], dd[i], seg) for i in near_exterior)
        #         )
        #         res = ma.get(1e100)
        #         ec[near_exterior, j] = np.array(res)
        print "Completed in %f seconds" % (time() - tic)
        return ec

    def compute_k(self, u, dd=None, dtheta=None, edge_corr=None, *args, **kwargs):
        """
        Compute anisotropic K in which distance is less than u and phi lies in the bins specified.
        :param u: Array of distances.
        :return: 2D array, rows represent values in u and cols represent between-values in phi
        """
        if not hasattr(u, '__iter__'):
            u = [u]
        dd = self.dd if dd is None else dd
        dtheta = self.dtheta if dtheta is None else dtheta
        edge_corr = self.edge_correction if edge_corr is None else edge_corr

        res = np.zeros((len(u), len(self.segments)))

        for j in range(self.nsegment):
            seg = self.segments[j]
            phi_ind = seg.contains(dtheta)
            # zero distance objects have a NaN angle change
            # we will apply these equally across ALL phi bins, so get an index here
            zero_d_ind = np.isnan(dtheta)
            for i in range(len(u)):
                t = u[i]
                d_ind = dd <= t
                # w = 1 / (edge_corr[d_ind & phi_ind] * phi_frac)  # using area-based correction
                w = 1 / edge_corr[d_ind & phi_ind]  # using area-based correction

                # compute contribution from zero distance links - these are spread evenly over all phi bins
                # w_zero = 1 / (edge_corr[d_ind & zero_d_ind] * phi_frac)
                w_zero = 1 / edge_corr[d_ind & zero_d_ind]

                a = (w.sum() + w_zero.sum() / float(self.nsegment))
                b = float(self.n * self.intensity)

                res[i, j] = a / b

        return res

    def run_permutation(self, u, niter=20):
        if np.any(u > self.max_d):
            raise AttributeError('No values of u may be > max_d')
        k = []
        try:
            for i in range(niter):
                data = CartesianData.from_args(*spatial.random_points_within_poly(self.domain, self.n))
                ii, jj, dd, dtheta, near_exterior = prepare_data(data, self.domain, self.max_d, compute_angles=True)
                ec = self.compute_edge_correction(data, ii, dd, self.segments, self.domain, self.n_quad)
                k.append(self.compute_k(u, dd=dd, dtheta=dtheta, edge_corr=ec))
        except Exception as exc:
            print repr(exc)
        finally:
            return np.array(k)


class RipleyKAnisotropicC2(RipleyKAnisotropic):
    """
    Similar to RipleyKAnisotropic, but the default phi bins are 'doubled' by adding in C2 rotational symmetry
    """
    def set_segments(self, segments=None):
        if segments is None:
            # revert to default behaviour
            dphi = np.pi / 4.
            phi0 = dphi / 2.
            phis = np.arange(phi0, np.pi, dphi)
            self.segments = [CircleSegment([p, p + np.pi], dphi, n_quad=self.n_quad) for p in phis]
        else:
            if not hasattr(segments, '__iter__'):
                segments = [segments]
            self.segments = segments


def phi_filter_factory(*args):
    """
    Create a filter function. Each pair of args defines a radial segment. The filter function represents the combined
    OR operation on these
    :param args: Pairs of phi values (a, b) enclosing segments, defined such that
    a <= phi < a + b
    :return: Filter function for phi.
    """
    assert len(args) % 2 == 0, "Phi threshold values must appear in pairs"
    assert len(args) >= 2, "Must provide at least two phi values to generate a filter function"
    for i in range(len(args) / 2):
        assert 0. <= args[i] <= 2 * np.pi, "Phi threshold lower value must be in the range [0, 2 * np.pi]"

    def filter_component(x, phi0, dphi):
        if phi0 + dphi < 2 * np.pi:
            return (phi0 <= np.mod(x, 2 * np.pi)) & (np.mod(x, 2 * np.pi) < phi0 + dphi)
        else:
            return (phi0 <= np.mod(x, 2 * np.pi)) | (np.mod(x, 2 * np.pi) < np.mod(phi0 + dphi, 2 * np.pi))

    def filter_func(x):
        phi0 = args[0]
        dphi = args[1]
        out = filter_component(x, phi0, dphi)
        for i in range(1, len(args) / 2):
            phi0 = args[2 * i]
            dphi = args[2 * i + 1]
            out = out | filter_component(x, phi0, dphi)
        return out

    return filter_func


def generate_angle_filters(phi, dphi):
    phi_width = [sum(t[1::2]) for t in phi]
    phi_filters = [phi_filter_factory(*t) for t in phi]
    return phi_filters, phi_width


def clock_plot(u, phi, k_obs, k_sim=None,
               title=None):
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    ax_ordering = [
        (0, 2),
        (0, 1),
        (0, 0),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
        (1, 2)
    ]
    running_max = 0
    for i in range(8):
        lhat = np.sqrt(k_obs[:, i] / np.pi)
        running_max = max(running_max, lhat.max())
        ax = axs[ax_ordering[i]]
        if k_sim is not None:
            y1 = np.sqrt(k_sim[:, :, i].min(axis=0) / np.pi)
            y2 = np.sqrt(k_sim[:, :, i].max(axis=0) / np.pi)
            ax.fill_between(u, y1, y2, facecolor='k', interpolate=True, alpha=0.4)
            running_max = max(running_max, y2.max())
        ax.plot(u, lhat, 'r-')
        if ax_ordering[i][0] != 2:
            ax.set_xticklabels([])
        if ax_ordering[i][1] != 0:
            ax.set_yticklabels([])

    # set running maximum ylim
    for i in range(8):
        ax = axs[ax_ordering[i]]
        ax.set_ylim([0, running_max * 1.02])

    # middle circle
    th = np.linspace(0, 2 * np.pi, 200)
    xc = np.cos(th)
    yc = np.sin(th)
    th = np.array([t[0] for t in phi])
    spokes = [np.array([np.linspace(0, 1, 50) * np.cos(t),
                        np.linspace(0, 1, 50) * np.sin(t)]) for t in th]
    axs[1, 1].plot(xc, yc, 'k-')
    [axs[1, 1].plot(t[0], t[1], 'k-') for t in spokes]
    axs[1, 1].set_xlim([-1.1, 1.1])
    axs[1, 1].set_ylim([-1.1, 1.1])
    axs[1, 1].set_aspect('equal')
    axs[1, 1].axis('off')

    big_ax = fig.add_subplot(111)
    big_ax.spines['top'].set_color('none')
    big_ax.spines['bottom'].set_color('none')
    big_ax.spines['left'].set_color('none')
    big_ax.spines['right'].set_color('none')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    big_ax.set_xlabel('Distance (m)')
    big_ax.set_ylabel('Anisotropic Ripley''s K')
    big_ax.patch.set_visible(False)

    plt.tight_layout(pad=1.5, h_pad=0.05, w_pad=0.05)
    big_ax.set_position([0.05, 0.05, 0.95, 0.9])
    if title:
        big_ax.set_title(title)


def anisotropy_array_plot(save=True, max_d=None, add_rose=True):
    """
    Not really a flexible method, more a way to isolate this plotting function
    If max_d is not specified, the full range is used.
    """
    OUTDIR = '/home/gabriel/Dropbox/research/output/ripley_anisotropy/'
    domains = chicago.get_chicago_side_polys(as_shapely=True)
    chicago_poly = chicago.compute_chicago_region(as_shapely=True).simplify(1000)

    CRIME_TYPES = (
        'burglary',
        'assault',
    )

    from matplotlib import pyplot as plt

    # Define phi combinations. This corresponds to inversion through the origin.
    combinations = [
        (3, 7),
        (0, 4),
        (1, 5),
        (2, 6),
    ]
    colours = ('k', 'k', 'r', 'r')
    linestyles = ('solid', 'dashed', 'solid', 'dashed')

    for ct in CRIME_TYPES:
        k_obs_dict = collections.defaultdict(dict)
        k_sim_dict = collections.defaultdict(dict)
        fig, axs = plt.subplots(3, 3, figsize=(10, 8))

        running_max = 0
        for i, r in enumerate(chicago_consts.REGIONS):
            rff = chicago_consts.FILE_FRIENDLY_REGIONS[r]
            ax_j = np.mod(i, 3)
            ax_i = i / 3
            ax = axs[ax_i, ax_j]
            infile = os.path.join(OUTDIR, 'ripley_%s_%s.pickle' % (rff, ct))
            with open(infile, 'r') as f:
                res = dill.load(f)
            k_obs_dict[ct][r] = res['k_obs']
            k_sim_dict[ct][r] = res['k_sim']
            u = res['u']
            if max_d is not None:
                idx = u <= max_d
                u = u[idx]
            for j, c in enumerate(combinations):
                ls = linestyles[np.mod(j, len(linestyles))]
                col = colours[np.mod(j, len(colours))]
                this_k_obs = res['k_obs'][:, c].sum(axis=1)
                if max_d is not None:
                    this_k_obs = this_k_obs[idx]
                running_max = max(running_max, this_k_obs.max())
                ax.plot(u, this_k_obs, c=col, linestyle=ls)
            # pick any simulated K - all angles look the same at this scale
            this_k_sim = res['k_sim'][:, :, :len(c)].sum(axis=2)
            if max_d is not None:
                this_k_sim = this_k_sim[:, idx]
            y0 = this_k_sim.min(axis=0)
            y1 = this_k_sim.max(axis=0)
            running_max = max(running_max, y1.max())
            ax.fill_between(u, y0, y1,
                            facecolor='k',
                            edgecolor='none',
                            alpha=0.4)

            if ax_i != 2:
                ax.set_xticklabels([])
            if ax_j != 0:
                ax.set_yticklabels([])


        for i in range(len(chicago_consts.REGIONS)):
            axs.flat[i].set_ylim([0, running_max * 1.02])
            # axs.flat[i].text(3, 0.9 * running_max, domain_mapping[REGIONS[i]])

        big_ax = fig.add_subplot(111)
        big_ax.spines['top'].set_color('none')
        big_ax.spines['bottom'].set_color('none')
        big_ax.spines['left'].set_color('none')
        big_ax.spines['right'].set_color('none')
        big_ax.set_xticks([])
        big_ax.set_yticks([])
        big_ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        big_ax.set_xlabel('Distance (m)')
        big_ax.set_ylabel('Anisotropic Ripley''s K')
        big_ax.patch.set_visible(False)

        plt.tight_layout(pad=1.5, h_pad=0.05, w_pad=0.05)
        big_ax.set_position([0.05, 0.05, 0.95, 0.9])

        inset_pad = 0.01  # proportion of parent axis width or height
        inset_width_ratio = 0.35
        inset_height_ratio = 0.55

        for i in range(len(chicago_consts.REGIONS)):
            ax_j = np.mod(i, 3)
            ax_i = i / 3
            ax = axs[ax_i, ax_j]
            ax_bbox = ax.get_position()
            inset_bbox = [
                ax_bbox.x0 + inset_pad * ax_bbox.width,
                ax_bbox.y1 - (inset_pad + inset_height_ratio) * ax_bbox.height,
                inset_width_ratio * ax_bbox.width,
                inset_height_ratio * ax_bbox.height,
            ]
            inset_ax = fig.add_axes(inset_bbox)
            chicago.plot_domain(chicago_poly, sub_domain=domains[chicago_consts.REGIONS[i]], ax=inset_ax)

        if add_rose:
            rose_width = 0.45
            phi = [((2 * i + 1) * np.pi / 8, np.pi / 4.) for i in range(8)]
            ax = axs[0, 2]  # top right axis
            ax_bbox = ax.get_position()
            inset_bbox = [
                ax_bbox.x1 - (inset_pad + rose_width) * ax_bbox.width,
                ax_bbox.y1 - (inset_pad + rose_width) * ax_bbox.height,
                rose_width * ax_bbox.width,
                rose_width * ax_bbox.height,
            ]
            inset_ax = fig.add_axes(inset_bbox, projection='polar')
            rose_plot(phi, combinations, ax=inset_ax)

        if save:
            fig.savefig('ripley_k_anisotropic_with_inset_%s.png' % ct, dpi=150)
            fig.savefig('ripley_k_anisotropic_with_inset_%s.pdf' % ct)


def rose_plot(phi, combinations, ax=None):
    from matplotlib import pyplot as plt
    colours = ('k', 'k', 'r', 'r')
    linestyles = ('solid', 'dashed', 'solid', 'dashed')

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
    th = np.linspace(0, 2 * np.pi, 500)
    r_circ = np.ones_like(th)
    ax.plot(th, r_circ)

    for i, c in enumerate(combinations):
        pff = lambda th: reduce(operator.__or__, (phi_filter_factory(*phi[j])(th) for j in c))
        this_segment = np.zeros_like(th)
        this_segment[pff(th)] = 1.
        ls = linestyles[i]
        col = colours[i]
        if ls == 'dashed':
            ax.fill_between(th, 0, this_segment,
                            facecolor='none',
                            edgecolor=col,
                            hatch='//\\')
        else:
            ax.fill_between(th, 0, this_segment,
                            facecolor=col,
                            edgecolor=col,
                            alpha=0.7)
    ax.set_yticks([])
    ax.set_xticks([])


if __name__ == '__main__':

    import os
    from scripts import OUT_DIR

    # OUTDIR = '/home/gabriel/Dropbox/research/output/'
    subdir = os.path.join(OUT_DIR, 'chicago', 'ripley_k_ani')
    if not os.path.isdir(subdir):
        os.makedirs(subdir)

    max_d = 500
    geos_simplification = 20  # metres tolerance factor
    n_sim = 20
    start_date = datetime.date(2011, 3, 1)
    end_date = start_date + datetime.timedelta(days=366)
    domains = chicago.get_chicago_side_polys(as_shapely=True)
    chicago_poly = chicago.compute_chicago_region(as_shapely=True).simplify(1000)

    # define a vector of threshold distances
    u = np.linspace(0, max_d, 2000)

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
        'chicago_southwest',
        'chicago_south',

        'chicago_far_southwest',
        'chicago_far_southeast',
        'chicago_west',

        'chicago_northwest',
        'chicago_north',
        'chicago_far_north',
    )

    CRIME_TYPES = (
        'burglary',
        'assault',
    )

    ## 1) run all Ripley's anisotropic

    res = collections.defaultdict(dict)

    for r in REGIONS:
        for ct in CRIME_TYPES:
            domain = domains[domain_mapping[r]].simplify(geos_simplification)
            if isinstance(domain, geometry.MultiPolygon):
                # quick fix for Far North, in which the first polygon is the vast majority of the region
                domain = domain[0]
            data, t0, cid = chicago.get_crimes_by_type(crime_type=ct,
                                                       start_date=start_date,
                                                       end_date=end_date,
                                                       domain=domain)
            tic = time()
            obj = RipleyKAnisotropicC2(data[:, 1:], max_d, domain)
            obj.process()
            k_obs = obj.compute_k(u)
            print "%s, %s, %f seconds" % (domain_mapping[r], ct, time() - tic)
            k_sim = obj.run_permutation(u, niter=n_sim)

            res[r][ct] = {
                'k_obs': k_obs,
                'k_sim': k_sim,
            }

            outfile = os.path.join(subdir, '%s_%s.pickle' % (r, ct))
            with open(outfile, 'w') as f:
                dill.dump(
                    {'obj': obj,
                     'k_obs': k_obs,
                     'k_sim': k_sim,
                     'u': u,
                     },
                    f
                )
            print "Completed %s %s" % (r, ct)
            del obj

    ## 2) Grid array of K_ani plots

    # from matplotlib import pyplot as plt
    #
    # combinations = [
    #     (3, 7),
    #     (0, 4),
    #     (1, 5),
    #     (2, 6),
    # ]
    # styles = ('k-', 'k--', 'r-', 'r--')
    #
    # for ct in CRIME_TYPES:
    #     k_obs_dict = collections.defaultdict(dict)
    #     k_sim_dict = collections.defaultdict(dict)
    #     fig, axs = plt.subplots(3, 3, figsize=(10, 8))
    #
    #     running_max = 0
    #     for i, r in enumerate(REGIONS):
    #         ax_i = np.mod(i, 3)
    #         ax_j = i / 3
    #         ax = axs[ax_i, ax_j]
    #         infile = os.path.join(OUTDIR, 'ripley_%s_%s.pickle' % (r, ct))
    #         with open(infile, 'r') as f:
    #             res = dill.load(f)
    #         k_obs_dict[ct][r] = res['k_obs']
    #         k_sim_dict[ct][r] = res['k_sim']
    #         for j, c in enumerate(combinations):
    #             this_k_obs = res['k_obs'][:, c].sum(axis=1)
    #             running_max = max(running_max, this_k_obs.max())
    #             ax.plot(res['u'], this_k_obs, styles[j])
    #
    #         # pick an arbitrary simulation combination - they are all the same
    #         this_k_sim = res['k_sim'][:, :, combinations[0]].sum(axis=2)
    #         ymax = this_k_sim.max(axis=0)
    #         ymin = this_k_sim.min(axis=0)
    #         running_max = max(running_max, ymax.max())
    #         ax.fill_between(res['u'], ymin, ymax, facecolor='k', alpha=0.4)
    #
    #         if ax_i != 2:
    #             ax.set_xticklabels([])
    #         if ax_j != 0:
    #             ax.set_yticklabels([])
    #
    #
    #     for i in range(len(REGIONS)):
    #         axs.flat[i].set_ylim([0, running_max * 1.02])
    #         # axs.flat[i].text(3, 0.9 * running_max, domain_mapping[REGIONS[i]])
    #
    #     big_ax = fig.add_subplot(111)
    #     big_ax.spines['top'].set_color('none')
    #     big_ax.spines['bottom'].set_color('none')
    #     big_ax.spines['left'].set_color('none')
    #     big_ax.spines['right'].set_color('none')
    #     big_ax.set_xticks([])
    #     big_ax.set_yticks([])
    #     big_ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    #     big_ax.set_xlabel('Distance (m)')
    #     big_ax.set_ylabel('Anisotropic Ripley''s K')
    #     big_ax.patch.set_visible(False)
    #
    #     plt.tight_layout(pad=1.5, h_pad=0.05, w_pad=0.05)
    #     big_ax.set_position([0.05, 0.05, 0.95, 0.9])
    #
    #     inset_pad = 0.01  # proportion of parent axis width or height
    #     inset_width_ratio = 0.35
    #     inset_height_ratio = 0.55
    #
    #     for i in range(len(REGIONS)):
    #         ax_i = np.mod(i, 3)
    #         ax_j = i / 3
    #         ax = axs[ax_i, ax_j]
    #         ax_bbox = ax.get_position()
    #         inset_bbox = [
    #             ax_bbox.x0 + inset_pad * ax_bbox.width,
    #             ax_bbox.y1 - (inset_pad + inset_height_ratio) * ax_bbox.height,
    #             inset_width_ratio * ax_bbox.width,
    #             inset_height_ratio * ax_bbox.height,
    #         ]
    #         inset_ax = fig.add_axes(inset_bbox)
    #         chicago.plot_domain(chicago_poly, sub_domain=domains[domain_mapping[REGIONS[i]]], ax=inset_ax)
    #
    #     fig.savefig('ripley_k_anisotropic_with_inset_%s.png' % ct, dpi=150)
    #     fig.savefig('ripley_k_anisotropic_with_inset_%s.pdf' % ct)

    ## 3) plot NORMED K_ani sim and obs

    # from matplotlib import pyplot as plt
    # max_u = 8000
    #
    # combinations = [
    #     (3, 7),
    #     (0, 4),
    #     (1, 5),
    #     (2, 6),
    # ]
    # colours = ('k', 'k', 'r', 'r')
    # linestyles = ('solid', 'dashed', 'solid', 'dashed')
    #
    # # k_obs_n = k_obs / k_obs[-1, :]
    # k_obs_n = k_obs
    # # k_sim_n = np.array([k_sim[i] / k_sim[i, -1, :] for i in range(k_sim.shape[0])])
    # k_sim_n = k_sim
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for i, c in enumerate(combinations):
    #     ls = linestyles[np.mod(i, len(linestyles))]
    #     col = colours[np.mod(i, len(colours))]
    #     this_k_obs_n = k_obs_n[:, c].sum(axis=1) / len(c)
    #     this_k_sim_n = k_sim_n[:, :, c].sum(axis=2) / len(c)
    #     y0 = this_k_sim_n.min(axis=0)
    #     y1 = this_k_sim_n.max(axis=0)
    #     idx = u <= max_u
    #     ax.plot(u[idx], this_k_obs_n[idx], c=col, linestyle=ls)
    #     if ls == 'dashed':
    #         ax.fill_between(u[idx], y0[idx], y1[idx],
    #                         facecolor='none',
    #                         edgecolor=col,
    #                         hatch='//\\')
    #     else:
    #         ax.fill_between(u[idx], y0[idx], y1[idx],
    #                         facecolor=col,
    #                         edgecolor=col,
    #                         alpha=0.3)
