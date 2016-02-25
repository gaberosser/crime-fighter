__author__ = 'gabriel'
import numpy as np
import math
from data.models import NetworkSpaceTimeData
from network.simulate import uniform_random_points_on_net, random_walk_normal
PI = np.pi


def exponential_decay_poisson(decay_const, intensity, t_max, prng=None):
    if not prng:
        prng = np.random.RandomState()

    def inv_func(t):
        return -math.log((intensity - t) / float(intensity)) / float(decay_const)

    tn = 0.
    res = []
    while True:
        tn -= math.log(prng.rand())
        try:
            ta = inv_func(tn)
            if ta <= t_max:
                res.append(ta)
            else:
                break
        except ValueError:
            # attempted to take the log of a negative number (outside of support of inv_func)
            break

    return np.array(res)


def nonstationary_poisson(inv_func, t_max, prng=None):
    if not prng:
        prng = np.random.RandomState()
    tn = 0.
    ta = 0.
    res = []
    while True:
        tn -= math.log(prng.rand())
        try:
            ta = inv_func(tn)
            if ta <= t_max:
                res.append(ta)
            else:
                break
        except ValueError:
            # attempted to take the log of a negative number (outside of support of inv_func)
            break

    return np.array(res)


def central_network_point(graph):
    xmin, ymin, xmax, ymax = graph.extent
    x = (xmax + xmin)/2.0
    y = (ymax + ymin)/2.0
    return graph.closest_edges_euclidean_brute_force(x, y)[0]


def network_centroid(graph):
    xmin, ymin, xmax, ymax = graph.extent
    x = (xmax + xmin) / 2.0
    y = (ymax + ymin) / 2.0
    return x, y


class SeppSimulationBase(object):
    def __init__(self, *args, **kwargs):

        # parameters
        # default values correspond to Mohler 2011 simulation

        self.t_total = kwargs.pop('t_total', None)  # to be set now or at runtime

        self.num_to_prune = 0  # to be set at runtime if required
        # data are set by run() method. Each datum has the format [idx, time, space, parent_idx]
        self._data = None  # set by run() method

        # BG and trigger parameters
        self.trigger_params = None
        self.bg_params = None

        self.set_bg_params(kwargs.pop('bg_params', None))
        self.set_trigger_params(kwargs.pop('trigger_params', None))

        self.prng = np.random.RandomState()

    @property
    def data(self):
        return [t[1:-1] for t in self._data]

    @property
    def bg_data(self):
        return [t[1:-1] for t in self._data if t[-1] is None]

    def set_bg_params(self, bg_params=None):
        """
        Parse supplied parameters or set defaults
        :param bg_params: Either a dictionary supplied at init or None
        :return:
        """
        raise NotImplementedError()

    def set_trigger_params(self, trigger_params=None):
        """
        Parse supplied parameters or set defaults
        :param trigger_params: Either a dictionary supplied at init or None
        :return:
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        self.prng.seed(seed)

    def bg_pdf(self, targets):
        raise NotImplementedError()

    def initialise_background(self):
        """ Simulate background events, setting parent idx as None """
        raise NotImplementedError()

    def spatial_excitation_one(self, s, size):
        """
        Generate the new spatial locations resulting from the excitation by the datum at location s.
        The time has already been generated.
        :param s: Location of parent
        :param size: Number of random draws to make. Much more efficient than drawing every time
        :return:
        """
        raise NotImplementedError()

    def temporal_excitation_one(self, t):
        """
        Generate the new times of offspring generated by a parent at time t
        :param t:
        :return: List of new times
        """
        raise NotImplementedError()

    def propagate_one(self, t, s, idx):
        """
        ** REPLACES _point_aftershocks **
        Generate the excitation series from a single datum with time t, location s and index idx.
        :param t:
        :param s:
        :param idx:
        :return: Array of new (time, space) data
        """
        # simulate triggered times
        triggered = []
        new_ts = self.temporal_excitation_one(t)
        # simulate triggered locations
        new_ss = self.spatial_excitation_one(s, new_ts.size)
        for tn, sn in zip(new_ts, new_ss):
            # append datum, minus the index, which is assigned by the parent process
            triggered.append([tn, sn, idx])
        return triggered

    def append_triggers(self):
        """
        Given the events in self.data, generate triggers and append
        :return: extended data array
        """
        i = 0  # rolling index

        while i < self.ndata:
            j, t, s, _ = self._data[i]
            if t > self.t_total:
                # increment index and skip this point - it is beyond the requested max simulation time
                i += 1
                continue

            new_events = self.propagate_one(t, s, j)
            if len(new_events) == 0:
                # no new events generated - increment index and skip this iteration
                i += 1
                continue

            # iterate, adding idx and include only new events within the requested maximum time
            idx = self.ndata
            for tn, sn, _ in new_events:
                if tn < self.t_total:
                    # the parent index will always be j
                    self._data.append([idx, tn, sn, j])
                    idx += 1

            # increment index
            i += 1

    def prune_and_relabel(self):
        """
        Prune the first and last set of points with size defined by n_prune.
        Then relabel the data to correct lineage IDs.
        Assume data are ordered by ascending time
        """

        self._data = self._data[self.num_to_prune:-self.num_to_prune]
        # relocate time so that first event occurs at t=0
        t0 = self._data[0][1]
        for t in self._data:
            t[1] -= t0

        # relabel triggered events after pruning
        # parent index is set to -1 if parent is no longer in the dataset
        parent_ids = set([t[0] for t in self._data])
        for t in self._data:
            the_link = t[-1]
            if t[-1] is not None and t[-1] not in parent_ids:
                # this parent no longer exists - it must have been pruned
                t[-1] = -1

    @property
    def ndata(self):
        return len(self._data)

    @property
    def number_bg(self):
        if self._data is None:
            raise AttributeError("Data have not been set.  Call run().")
        return len([t[-1] for t in self._data if t[-1] is None])

    @property
    def number_trigger(self):
        if self._data is None:
            raise AttributeError("Data have not been set.  Call run().")
        return len([t[-1] for t in self._data if t[-1] is not None])

    @property
    def linkages(self):
        """
        :return: bg_idx, cause_idx, effect_idx. True linkages in simulation.
        """
        # find trigger events with parents still in the dataset
        linked_map = [t[-1] is not None and t[-1] != -1 for t in self._data]
        id_to_idx = dict([
            (t[0], i) for i, t in enumerate(self._data)
        ])
        effect_idx = [i for i, t in enumerate(linked_map) if t]
        cause_idx = [id_to_idx[self._data[i][-1]] for i in effect_idx]
        bg_idx = [i for i, t in enumerate(self._data) if t[-1] is None]

        return bg_idx, cause_idx, effect_idx

    @property
    def p(self):
        """ True probability matrix for the simulated data """
        p = np.zeros((self.ndata, self.ndata), dtype=np.bool)
        bg_idx, cause_idx, effect_idx = self.linkages

        # insert links
        p[bg_idx, bg_idx] = True
        p[cause_idx, effect_idx] = True

        return p

    def run(self, t_total=None, num_to_prune=None):
        self.t_total = t_total or self.t_total
        if not self.t_total:
            raise ValueError("The value of t_total has not been set.")

        self.num_to_prune = num_to_prune or self.num_to_prune
        # set background events
        self.initialise_background()
        # iterate over the growing data array to create branching aftershocks
        self.append_triggers()

        # sort by time and reindex
        t = np.array([t[1] for t in self._data])
        sort_idx = t.argsort()

        self._data = [self._data[i] for i in sort_idx]
        if self.num_to_prune:
            self.prune_and_relabel()


class PlanarGaussianSpaceExponentialTime(SeppSimulationBase):
    """
    BG: sum of gaussians
    Trigger: single gaussian in space, exponential in time
    """
    @property
    def default_bg_params(self):
        return [{
            'location': [0., 0.],
            'intensity': 5.71,  # events day^-1
            'sigma': [4.5, 4.5]
        }]

    @property
    def default_trigger_params(self):
        return {
            'time_decay': 0.1,  # day^-1
            'intensity': 0.2,
            'sigma': [0.01, 0.1],
        }

    @property
    def data(self):
        # cast to np array for convenience
        return np.array([[t[1], t[2][0], t[2][1]] for t in self._data])

    @property
    def bg_data(self):
        return np.array([[t[1], t[2][0], t[2][1]] for t in self._data if t[-1] is None])

    # convenient properties
    @property
    def trigger_decay(self):
        return self.trigger_params['time_decay']

    @property
    def trigger_intensity(self):
        return self.trigger_params['intensity']

    def set_bg_params(self, bg_params=None):
        """
        Parse supplied parameters or set defaults
        :param bg_params: Either a dictionary supplied at init or None
        :return:
        """
        exp_keys = ('location', 'intensity', 'sigma')
        if bg_params is None:
            bg_params = self.default_bg_params
        elif isinstance(bg_params, dict):
            bg_params = [bg_params]
        for t in bg_params:
            for ek in exp_keys:
                assert ek in t, "Each gaussian BG component must have the attribute '%s'" % ek
        self.bg_params = bg_params

    def set_trigger_params(self, trigger_params=None):
        """
        Parse supplied parameters or set defaults
        :param trigger_params: Either a dictionary supplied at init or None
        :return:
        """
        exp_keys = ('time_decay', 'intensity', 'sigma')
        if trigger_params is None:
            self.trigger_params = self.default_trigger_params
        else:
            for ek in exp_keys:
                assert ek in trigger_params, "Trigger parameters must have the attribute '%s'" % ek
            self.trigger_params = trigger_params

    def trigger_cov(self, *args):
        return np.diag(self.trigger_params['sigma']) ** 2

    def bg_pdf(self, targets):
        raise NotImplementedError()

    def initialise_background(self):
        """ Simulate background events, setting parent idx as None """
        data = []
        count = 0
        for bg in self.bg_params:
            number_bg = self.prng.poisson(bg['intensity'] * self.t_total)
            # background event times uniform on interval
            this_times = self.prng.rand(number_bg) * self.t_total
            # background locations distributed according to bg_params
            this_locations = self.prng.multivariate_normal(bg['location'], np.diag(bg['sigma']) ** 2, number_bg)
            for t, s in zip(this_times, this_locations):
                data.append([count, t, s, None])
                count += 1
        self._data = data

    def temporal_excitation_one(self, t):
        new_ts = exponential_decay_poisson(self.trigger_decay,
                                           self.trigger_intensity,
                                           self.t_total - t,
                                           prng=self.prng)
        # add existing time back on
        new_ts += t
        return new_ts

    def spatial_excitation_one(self, s, n):
        """
        Generate the new spatial location resulting from the excitation by the supplied datum.
        The time has already been generated.
        :return:
        """
        # covariance matrix may be a function of the location
        cov = self.trigger_cov(s)
        return self.prng.multivariate_normal(s, cov, size=n)


class MohlerSimulation(PlanarGaussianSpaceExponentialTime):

    def __init__(self, **kwargs):

        super(MohlerSimulation, self).__init__(**kwargs)
        self.t_total = 1284
        self.num_to_prune = 2000


class HomogPoissonBackgroundSimulation(PlanarGaussianSpaceExponentialTime):

    @property
    def bg_intensity(self):
        return self.bg_params['intensity']

    def set_bg_params(self, bg_params=None):
        """
        Parse supplied parameters or set defaults
        :param bg_params: Either a dictionary supplied at init or None
        :return:
        """
        exp_keys = ('intensity', 'xmin', 'xmax', 'ymin', 'ymax')
        if bg_params is None:
            bg_params = self.default_bg_params
        else:
            for ek in exp_keys:
                assert ek in bg_params, "BG params dictionary must have the attribute '%s'" % ek
        self.bg_params = bg_params

    @property
    def default_bg_params(self):
        return {
            'intensity': 5,
            'xmin': -10,
            'xmax': 10,
            'ymin': -10,
            'ymax': 10,
        }

    def initialise_background(self):
        """ Simulate background events as a bounded homogeneous Poisson process """
        bg = self.bg_params
        number_bg = self.prng.poisson(self.bg_intensity * self.t_total)
        # background event times uniform on interval
        this_times = self.prng.rand(number_bg) * self.t_total
        # background locations distributed according to bg_params
        x_locs = self.prng.rand(number_bg) * (bg['xmax'] - bg['xmin']) + bg['xmin']
        y_locs = self.prng.rand(number_bg) * (bg['ymax'] - bg['ymin']) + bg['ymin']
        self._data = [[i, t, (x, y), None] for (i, t), x, y in zip(enumerate(this_times), x_locs, y_locs)]


class PatchyGaussianSumBackground(PlanarGaussianSpaceExponentialTime):

    @property
    def default_bg_params(self):
        return [
            {
                'location': [-10, -10],
                'intensity': 1.,
                'sigma': [5., 5.],
            },
            {
                'location': [-10, 10],
                'intensity': 1.,
                'sigma': [5., 5.],
            },
            {
                'location': [10, -10],
                'intensity': 1.,
                'sigma': [5., 5.],
            },
            {
                'location': [10, 10],
                'intensity': 1.,
                'sigma': [5., 5.],
            }
        ]

    @property
    def default_trigger_params(self):
        return {
            'intensity': 0.2,  # events day^-1
            'time_decay': 0.1,  # day^-1
            'sigma': [.01, .1]
        }


class LocalTriggeringSplitByQuartiles(PlanarGaussianSpaceExponentialTime):

    @property
    def default_trigger_params(self):
        def sigma_fun(s):
            x, y = s
            if x > 0 and y > 0:
                return [0.01, 0.1]
            if x > 0 and y <= 0:
                return [0.1, 0.01]
            if x <= 0 and y <= 0:
                return [0.01, 0.01]
            if x <= 0 and y > 0:
                return [0.1, 0.1]

        return {
            'intensity': 0.2,
            'time_decay': 0.1,  # day^-1
            'sigma': sigma_fun,
        }

    def trigger_cov(self, s, *args):
        sigma = self.trigger_params['sigma'](s)
        return np.diag(sigma) ** 2


class NetworkHomogBgExponentialGaussianTrig(PlanarGaussianSpaceExponentialTime):
    """
    Network-constrained SEPP
    BG is uniform on the network
    Triggering is a gaussian random walk on the network and exponentially decaying over time
    """
    def __init__(self, net, **kwargs):
        self.net = net
        super(NetworkHomogBgExponentialGaussianTrig, self).__init__(**kwargs)

    def set_bg_params(self, bg_params=None):
        """
        Parse supplied parameters or set defaults
        :param bg_params: Either a dictionary supplied at init or None
        :return:
        """
        exp_keys = ('intensity',)
        if bg_params is None:
            bg_params = self.default_bg_params
        else:
            for ek in exp_keys:
                assert ek in bg_params, "BG params dictionary must have the attribute '%s'" % ek
        self.bg_params = bg_params

    @property
    def default_bg_params(self):
        return {
            'intensity': 5.,  # events day^-1
        }

    @property
    def default_trigger_params(self):
        return {
            'time_decay': 0.1,  # day^-1
            'intensity': 0.2,
            'sigma': 0.1,
        }

    @property
    def bg_intensity(self):
        return self.bg_params['intensity']

    def trigger_sigma(self, s):
        return self.trigger_params['sigma']

    @property
    def data(self):
        # cast to NetworkData array for convenience
        return NetworkSpaceTimeData([[t[1], t[2]] for t in self._data])

    @property
    def bg_data(self):
        return NetworkSpaceTimeData([[t[1], t[2]] for t in self._data if t[-1] is None])

    def bg_pdf(self, targets):
        raise NotImplementedError()

    def initialise_background(self):
        """ Simulate background events, setting parent idx as None """
        data = []
        count = 0
        number_bg = self.prng.poisson(self.bg_intensity * self.t_total)
        # background event times uniform on interval
        this_times = self.prng.rand(number_bg) * self.t_total
        # background locations distributed uniformly
        this_locations = uniform_random_points_on_net(self.net, n=len(this_times))
        for t, s in zip(this_times, this_locations.toarray()):
            data.append([count, t, s, None])
            count += 1
        self._data = data

    def spatial_excitation_one(self, s, n):
        """
        Generate the new spatial location resulting from the excitation by the supplied datum.
        The time has already been generated.
        :return:
        """
        # covariance matrix may be a function of the location
        sigma = self.trigger_sigma(s)
        # TODO: can improve the efficiency here by supporting multiple draws in the random_walk_normal method
        return [random_walk_normal(s, sigma=sigma) for i in range(n)]