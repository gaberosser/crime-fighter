__author__ = 'gabriel'
import numpy as np
import math
import random
import datetime
import pandas
PI = np.pi

def nonstationary_poisson(inv_func, t_max):
    tn = 0.
    ta = 0.
    res = []
    while True:
        tn -= math.log(random.random())
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


class MohlerSimulation():

    def __init__(self):

        # parameters
        self.t_total = 1284 # time extent being considered (s)
        self.number_to_prune = 2000 # number particles to prune at start and end

        self.bg_mu_bar = 5.71 # events / (m^2 s)
        self.bg_sigma = 4.5 # spatial width of background centered gaussian

        self.off_theta = 0.2 # aftershock strength param
        self.off_omega = 0.1 # aftershock time decay constant (s^-1)
        self.off_sigma_x = 0.01 # aftershock x width (m)
        self.off_sigma_y = 0.1 # aftershock y width (m)

        self.off_mean = np.zeros(2)

        """ May need a normalisation constant in the aftershock process for direct comparison with Mohler,
            as the authors don't use a normalised distribution in that case. """
        K = 1/float(2 * PI * self.off_sigma_x * self.off_sigma_y)

        self.data = np.zeros((1, 4))

    def data_iter(self):
        i = 0
        while i<self.ndata:
            yield self.data[i, :]
            i += 1


    @property
    def off_cov(self):
        off_var_x = self.off_sigma_x ** 2
        off_var_y = self.off_sigma_y ** 2
        return np.array([[off_var_x, 0.], [0., off_var_y]])

    @property
    def bg_var(self):
        return self.bg_sigma ** 2

    def non_stationary_poisson_inverse(self, x):
        # inverse function for simulating non-stationary poisson process
        return -1.0 / float(self.off_omega) * math.log((self.off_theta - x)/self.off_theta)

    def initialise_background(self):
        """ Simulate background events
            NB this destroys all existing data. """
        number_bg = np.random.poisson(self.bg_mu_bar * self.t_total)
        # background event times uniform on interval
        bg_times = np.random.random(number_bg) * self.t_total
        # background locations distributed normally
        bg_locations = np.random.multivariate_normal(np.zeros(2), np.array([[self.bg_var, 0.], [0., self.bg_var]]), number_bg)
        self.data = np.array([[i, t, x, y, np.nan] for (i, t), (x, y) in zip(enumerate(bg_times), bg_locations)])
        return bg_times, bg_locations

    def _point_aftershocks(self, t, x, y, idx):
        """ Generate sequence of aftershocks for the given shock t, x, y datum.
            Shocks are appended to list """
        # generate further points
        t_max = self.t_total
        new_t = nonstationary_poisson(self.non_stationary_poisson_inverse, self.t_total - t)
        shocks = []
        loc = np.random.multivariate_normal(self.off_mean, self.off_cov, size=len(new_t))
        for tn, ln in zip(new_t, loc):
            shocks.append([tn, ln[0], ln[1]])
            self.data = np.vstack((self.data, [len(self.data), t + tn, x + ln[0], y + ln[1], idx]))
        return shocks

    def generate_aftershocks(self):
        shocks = []
        n_init = self.data.shape[0]
        n_shocks = 0
        gen = self.data_iter()
        for (i, t, x, y, _) in gen:
            print "%u: %u / %u" % (i, len(self.data), n_init)
            new_shocks = self._point_aftershocks(t, x, y, i)
            shocks.extend(new_shocks)
            if i < n_init:
                n_shocks += len(new_shocks)
        # TODO: test that the statement below matches expectation of aftershock generation strength
        # print "Generated %d shocks from the %d background shocks" % (n_shocks, n_init)
        return shocks

    def prune_points(self, n_prune):
        """ Prune the first and last set of points with size defined by n_prune. """
        # assume data are in a sorted np array
        self.data = self.data[n_prune:-n_prune, :]
        # relocate time so that first event occurs at t=0
        t0 = self.data[0, 1]
        self.data[:, 1] -= t0

    def relabel(self):
        # relabel aftershocks with removed parents
        # relabel all indices
        parent_ids = self.data[:, 0].astype(int)
        ash_idx = np.where(~np.isnan(self.data[:, 4]))[0]
        link_ids = self.data[ash_idx, 4].astype(int)
        for i in range(link_ids.size):
            this_link_id = link_ids[i]
            if not np.any(parent_ids == this_link_id):
                self.data[ash_idx[i], 4] = -1.
            else:
                new_idx = np.where(parent_ids == this_link_id)[0]
                if len(new_idx) != 1:
                    raise ValueError("Duplicate ID found")
                self.data[ash_idx[i], 4] = new_idx[0]
        self.data = self.data[:, 1:]

    @property
    def ndata(self):
        return self.data.shape[0]

    @property
    def number_bg(self):
        return np.sum(np.isnan(self.data[:, -1]))

    @property
    def number_aftershocks(self):
        return np.sum(~np.isnan(self.data[:, -1]))

    @property
    def p(self):
        """ True probability matrix for the simulated data """
        p = np.zeros((self.ndata, self.ndata))
        linked_map = (~np.isnan(self.data[:, -1])) & (self.data[:, -1] != -1)
        link_idx = np.where(linked_map)[0]
        bg_idx = np.where(~linked_map)[0]

    def run(self):
        self.initialise_background()
        # iterate over the growing data array to create branching aftershocks
        self.generate_aftershocks()

        self.data = np.array(self.data)
        # sort by time and reindex
        sort_idx = self.data[:, 1].argsort()

        self.data = self.data[sort_idx]
        self.prune_points(self.number_to_prune)
        self.relabel()

