__author__ = 'gabriel'
import numpy as np
import datetime
import pandas
PI = np.pi

def nonstationary_poisson(inv_func, upper, t_max):
    tn = 0.
    ta = 0.
    res = []
    while True:
        tn -= np.log(np.random.random())
        if tn < upper:
            ta = inv_func(tn)
            if ta <= t_max:
                res.append(ta)
            else:
                break
        else:
            break
    return np.array(res)


class MohlerSimulation():

    def __init__(self):

        # parameters
        self.t_total = 1050 # time extent being considered (s)
        self.number_to_prune = 2000 # number particles to prune at start and end

        self.bg_mu_bar = 5.71 # events / (m^2 s)
        self.bg_sigma = 4.5 # spatial width of background centered gaussian

        self.off_theta = 0.5 # aftershock strength param
        self.off_omega = 0.1 # aftershock time decay constant (s^-1)
        self.off_sigma_x = 0.01 # aftershock x width (m)
        self.off_sigma_y = 0.1 # aftershock y width (m)

        self.bg_var = np.power(self.bg_sigma, 2)
        self.off_var_x = np.power(self.off_sigma_x, 2)
        self.off_var_y = np.power(self.off_sigma_y, 2)
        self.off_mean = np.zeros(2)
        self.off_cov = np.array([[self.off_var_x, 0.], [0., self.off_var_y]])

        """ May need a normalisation constant in the aftershock process for direct comparison with Mohler,
            as the authors don't use a normalised distribution in that case. """
        K = 1/float(2 * PI * self.off_sigma_x * self.off_sigma_y)

        # inverse function for simulating non-stationary poisson process
        self.inv_func = lambda a: -1/float(self.off_omega) * np.log(np.abs((self.off_theta - a)/self.off_theta))
        self.upper = self.off_theta

        self.data = []

    def initialise_background(self):
        """ Simulate background events
            NB this destroys all existing data. """
        number_bg = np.random.poisson(self.bg_mu_bar * self.t_total)
        # background event times uniform on interval
        bg_times = np.random.random(number_bg) * self.t_total
        # background locations distributed normally
        bg_locations = np.random.multivariate_normal(np.zeros(2), np.array([[self.bg_var, 0.], [0., self.bg_var]]), number_bg)
        self.data = [[t, x, y, True] for t, (x, y) in zip(bg_times, bg_locations)]
        return bg_times, bg_locations

    def generate_aftershocks(self, t, x, y):
        """ Generate sequence of aftershocks for the given shock t, x, y datum.
            Shocks are appended to list """
        # generate further points
        t_max = self.t_total
        new_t = nonstationary_poisson(self.inv_func, self.upper, self.t_total - t)
        loc = np.random.multivariate_normal(self.off_mean, self.off_cov, size=len(new_t))
        for tn, ln in zip(new_t, loc):
            self.data.append([t + tn, x + ln[0], y + ln[1], False])
        return len(new_t)

    def prune_points(self, n_prune):
        """ Prune the first and last set of points with size defined by n_prune. """
        # assume data are sorted
        del self.data[:n_prune]
        del self.data[-n_prune:]

    def run(self):
        self.initialise_background()
        for t, x, y, _ in self.data:
            self.generate_aftershocks(t, x, y)
        # sort by time
        self.data = sorted(self.data, key=lambda x: x[0])
        self.prune_points(self.number_to_prune)

