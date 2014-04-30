__author__ = 'gabriel'
import estimation, simulate
from kde.methods import pure_python as pp_kde
import numpy as np

c = simulate.MohlerSimulation()
c.run()
data = np.array(c.data)[:, :3] # (t, x, y, b_is_BG)
P = estimation.initial_guess(data)
sample_idx = estimation.sample_events(P)
bg = []
interpoint = []
for x0, x1 in sample_idx:
    if x0 == x1:
        # bg
        bg.append(data[x0, :])
    else:
        # offspring
        dest = data[x0, :]
        origin = data[x1, :]
        interpoint.append(dest - origin)
bg = np.array(bg)
interpoint = np.array(interpoint)
bg_t_kde = pp_kde.VariableBandwidthKde(bg[:, 0])
bg_xy_kde = pp_kde.VariableBandwidthKde(bg[:, 1:])
trigger_kde = pp_kde.VariableBandwidthKde(interpoint)