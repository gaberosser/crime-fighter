__author__ = 'gabriel'

import estimation, simulate
from kde.methods import pure_python as pp_kde
import numpy as np

print "Starting simulation..."
# simulate data
c = simulate.MohlerSimulation()
c.bg_mu_bar = 1.0
c.number_to_prune = 100
c.run()
data = np.array(c.data)[:, :3]  # (t, x, y, b_is_BG)
ndata = data.shape[0]
# sort data by time ascending (may be done already?)
data = data[data[:, 0].argsort()]
print "Complete"


# compute pairwise (t, x, y) difference vector
t1, t2 = np.meshgrid(data[:,0], data[:,0])
td = t1 - t2
del t1, t2
x1, x2 = np.meshgrid(data[:,1], data[:,1])
xd = x1 - x2
del x1, x2
y1, y2 = np.meshgrid(data[:,2], data[:,2])
yd = y1 - y2
del y1, y2
pdiff = np.dstack((td, xd, yd))
del td, xd, yd

# if the system memory is limited:
# pdiff = np.zeros((ndata, ndata, 3))
# for i in range(ndata):
#     for j in range(i):
#         pdiff[j, i, :] = data[i] - data[j]

print "Initial estimate of P_0..."
P = estimation.initial_guess(pdiff)
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
print "Complete"

print "Computing KDEs..."
# compute KDEs
# BG
bg_t_kde = pp_kde.VariableBandwidthKde(bg[:, 0])
bg_xy_kde = pp_kde.VariableBandwidthKde(bg[:, 1:])
# interpoint / trigger KDE
trigger_kde = pp_kde.VariableBandwidthKde(interpoint)
print "Complete"

print "Evaluating BG KDE..."
# evaluate BG at data points
m_xy = bg_xy_kde.pdf(data[:, 1], data[:, 2])
m_t = bg_t_kde.pdf(data[:, 0])
m = m_xy * m_t
print "Complete"

print "Evaluating trigger KDE..."
# evaluate trigger KDE
I, J, G = estimation.evaluate_trigger_kde(trigger_kde, data)
g = np.zeros_like(P)
g[I, J] = G
del I, J, G
print "Complete"


print "Computing lambda..."
l = np.sum(g, axis=1) + m


print "Computing P_1..."
P = np.zeros_like(P)
