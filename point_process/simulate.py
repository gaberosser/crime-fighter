__author__ = 'gabriel'
import numpy as np
import math
PI = np.pi

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


class SeppSimulation(object):

    def __init__(self, **kwargs):

        # parameters
        # default values correspond to Mohler 2011 simulation

        self.t_total = kwargs.pop('t_total', None)  # to be set now or at runtime

        self.num_to_prune = 0  # to be set at runtime if required
        self.data = None  # set by run() method

        # BG process(es): array of dictionaries, one per BG hotspot
        bg_params = kwargs.pop('bg_params', None)
        if bg_params is not None and hasattr(bg_params, '__iter__'):
            self.bg_params = bg_params
        elif bg_params is not None and isinstance(bg_params, dict):
            self.bg_params = [bg_params]
        else:
            self.bg_params = self.default_bg_params

        trigger_params = kwargs.pop('trigger_params', None)
        if trigger_params is not None and isinstance(trigger_params, dict):
            self.trigger_params = trigger_params
        else:
            self.trigger_params = self.default_trigger_params

        self.prng = np.random.RandomState()

    @property
    def default_bg_params(self):
        return [{
                'location': [0., 0.],
                'intensity': 5.71,  # events day^-1
                'sigma': [4.5, 4.5],
                }]

    @property
    def default_trigger_params(self):
        return {
            'intensity': 0.2,
            'time_decay': 0.1,  # day^-1
            'sigma': [0.01, 0.1]
        }

    def seed(self, seed=None):
        self.prng.seed(seed)

    @property
    def trigger_cov(self):
        trigger_var = np.array(self.trigger_params['sigma']) ** 2
        return np.diag(trigger_var)

    def non_stationary_poisson_inverse(self, x):
        # inverse function for simulating non-stationary poisson process
        return -1.0 / float(self.trigger_params['time_decay']) * math.log((self.trigger_params['intensity'] - x) /
                                                                          self.trigger_params['intensity'])

    def initialise_background(self):
        """ Simulate background events """
        data = []
        count = 0
        for bg in self.bg_params:
            number_bg = self.prng.poisson(bg['intensity'] * self.t_total)
            # background event times uniform on interval
            this_times = self.prng.rand(number_bg) * self.t_total
            # background locations distributed according to bg_params
            this_locations = self.prng.multivariate_normal(bg['location'], np.diag(np.array(bg['sigma']) ** 2), number_bg)
            data.append(np.array([[count + i, t, x, y, np.nan] for (i, t), (x, y) in zip(enumerate(this_times), this_locations)]))
            count += this_times.size
        return np.vstack(tuple(data))

    def _point_aftershocks(self, t, x, y, idx):
        """ Generate sequence of triggered events for the given (t, x, y) datum.
            Events are appended to a list. NB They are not given an index - this should be done in the main
             calling routine """

        t_max = self.t_total
        new_t = nonstationary_poisson(self.non_stationary_poisson_inverse, self.t_total - t, prng=self.prng)
        triggered = []
        loc = self.prng.multivariate_normal(np.array([x, y]), self.trigger_cov, size=len(new_t))
        for tn, xn in zip(new_t, loc):
            triggered.append([t + tn, xn[0], xn[1], idx])
        return np.array(triggered)

    def append_triggers(self, data):
        """
        Given the events in data, generate triggers and append to data.  Return the new array
        :param data: e.g. from initialise_background
        :return: extended data array
        """

        i = 0  # rolling index

        while i < data.shape[0]:
            j, t, x, y, _ = data[i]
            if t > self.t_total:
                # increment index and skip this point - it is beyond the requested max simulation time
                i += 1
                continue

            new_events = self._point_aftershocks(t, x, y, j)
            if new_events.size == 0:
                # no new events generated - increment index and skip this iteration
                i += 1
                continue

            # include only new events within the requested maximum time
            new_events = new_events[new_events[:, 0] < self.t_total]
            if new_events.size == 0:
                # no new events left - increment index and skip this iteration
                i += 1
                continue

            n_new = new_events.shape[0]

            # prepend indices to new events
            new_events = np.hstack((np.arange(data.shape[0], data.shape[0] + n_new).reshape((n_new, 1)), new_events))

            # append new_events to running list
            # triggered = np.vstack((triggered, new_events))

            # append new events to data array
            data = np.vstack((data, new_events))

            # increment index
            i += 1

        return data

    def prune_and_relabel(self, n_prune):
        """ Prune the first and last set of points with size defined by n_prune.
            Then relabel the data to correct lineage IDs. """

        # assume data are in a sorted np array
        self.data = self.data[n_prune:-n_prune, :]
        # relocate time so that first event occurs at t=0
        t0 = self.data[0, 1]
        self.data[:, 1] -= t0

        # relabel triggered events after pruning
        # parent index is set to -1 if parent is no longer in the dataset

        parent_ids = self.data[:, 0].astype(int)
        trigger_idx = np.where(~np.isnan(self.data[:, 4]))[0]
        link_ids = self.data[trigger_idx, 4].astype(int)

        for i in range(link_ids.size):
            this_link_id = link_ids[i]
            # search for corresponding parent
            if not np.any(parent_ids == this_link_id):
                # parent no longer present in dataset
                self.data[trigger_idx[i], 4] = -1.
            else:
                # parent is present, update index
                new_idx = np.where(parent_ids == this_link_id)[0]
                if len(new_idx) != 1:
                    raise ValueError("Duplicate ID found")
                self.data[trigger_idx[i], 4] = new_idx[0]

    @property
    def ndata(self):
        return self.data.shape[0]

    @property
    def number_bg(self):
        return np.sum(np.isnan(self.data[:, -1]))

    @property
    def number_trigger(self):
        return np.sum(~np.isnan(self.data[:, -1]))

    @property
    def linkages(self):
        """
        :return: bg_idx, cause_idx, effect_idx. True linkages in simulation.
        """
        # find trigger events with parents still in the dataset
        linked_map = (~np.isnan(self.data[:, -1])) & (self.data[:, -1] != -1)

        # extract indices
        effect_idx = np.where(linked_map)[0]
        cause_idx = self.data[effect_idx, -1]
        bg_idx = np.where(~linked_map)[0]

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
        data = self.initialise_background()
        # iterate over the growing data array to create branching aftershocks
        data = self.append_triggers(data)

        # sort by time and reindex
        sort_idx = data[:, 1].argsort()

        self.data = data[sort_idx]
        if self.num_to_prune:
            self.prune_and_relabel(self.num_to_prune)

        self.data = self.data[:, 1:]



class MohlerSimulation(SeppSimulation):

    @property
    def default_bg_params(self):
        return [{
                'location': [0., 0.],
                'intensity': 5.71,  # events day^-1
                'sigma': [4.5, 4.5],
                }]

    @property
    def default_trigger_params(self):
        return {
            'intensity': 0.2,
            'time_decay': 0.1,  # day^-1
            'sigma': [0.01, 0.1]
        }

    def __init__(self, **kwargs):

        super(MohlerSimulation, self).__init__(**kwargs)
        self.t_total = 1284
        self.num_to_prune = 2000


class MySimulation1(SeppSimulation):

    @property
    def default_bg_params(self):
        gen_obj = lambda pos: {
            'location': pos,
            'intensity': 1.25,  # events day^-1
            'sigma': [.25, .25],
        }
        return [
            gen_obj([-1., -1.]), gen_obj([-1., 1.]), gen_obj([1., -1.]), gen_obj([1., 1.])
        ]

    @property
    def default_trigger_params(self):
        return {
            'intensity': 0.2,  # events day^-1
            'time_decay': 0.1,  # day^-1
            'sigma': [.01, .01]
        }