__author__ = 'gabriel'
import numpy as np
import datetime
import pandas

# parameters
t_extent = 5000 # time extent being considered (s)
t_buffer = 2000 # number of initial and final points to discard

bg_mu_bar = 5.71 # events / (m^2 s)
bg_sigma = 4.5 # spatial width of background centered gaussian

off_theta = 0.5 # aftershock strength param
off_omega = 0.1 # aftershock time decay constant (s^-1)
off_sigma_x = 0.01 # aftershock x width (m)
off_sigma_y = 0.1 # aftershock y width (m)

t_total = t_extent + 2 * t_buffer
bg_var = np.power(bg_sigma, 2)
off_var_x = np.power(off_sigma_x, 2)
off_var_y = np.power(off_sigma_y, 2)

""" May need a normalisation constant in the aftershock process for direct comparison with Mohler,
    as the authors don't use a normalised distribution in that case. """
K = 1/float(2 * np.pi * off_sigma_x * off_sigma_y)

# simulate background events
number_bg = np.random.poisson(bg_mu_bar * t_total)
# background event times uniform on interval
bg_times = np.random.random(number_bg) * t_total
# background locations distributed normally
bg_locations = np.random.multivariate_normal(np.zeros(2), np.array([[bg_var, 0.], [0., bg_var]]), number_bg)

# inverse function for simulating non-stationary poisson process
inv_func = lambda a: -1/float(off_omega) * np.log(np.abs((off_theta - a)/off_theta))

# simulate aftershock events
off_times = []
off_locations = []
for i in range(number_bg):
    x = bg_locations[i, 0]
    y = bg_locations[i, 1]
    t = bg_times[i]
    tn = 0. # standardized arrival time
    ta = t
    off_count = 0
    while True:
        u = np.random.random()
        tn -= np.log(u)
        ta += inv_func(tn) if tn < off_theta else (t_extent + t_buffer + 1.)
        if ta < (t_extent + t_buffer):
            off_count += 1
            # generate offspring location
            xn, yn = np.random.multivariate_normal(np.zeros(2), np.array([[off_var_x, 0.], [0., off_var_y]]))
            x += xn
            y += yn
            off_times.append(ta)
            off_locations.append([x, y])
        else:
            print "Completed particle %u: %u aftershocks" % (i, off_count)
            break
