__author__ = 'gabriel'
import estimation, simulate
from kde.methods import pure_python as pp_kde
import numpy as np

# simulate data
c = simulate.MohlerSimulation()
c.run()
data = np.array(c.data)[:, :3]  # (t, x, y, b_is_BG)
ndata = data.shape[0]
# sort data by time ascending (may be done already?)
data = data[data[:, 0].argsort()]

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

# compute KDEs
# BG
bg_t_kde = pp_kde.VariableBandwidthKde(bg[:, 0])
bg_xy_kde = pp_kde.VariableBandwidthKde(bg[:, 1:])
# interpoint / trigger KDE
trigger_kde = pp_kde.VariableBandwidthKde(interpoint)

# evaluate BG at grid points
m_xy = bg_xy_kde.values_at_data()
m_t = bg_t_kde.values_at_data()
m = m_xy * m_t

# evaluate trigger KDE
I, J, G = estimation.evaluate_trigger_kde(trigger_kde, data)

# max_t, max_d = estimation.compute_trigger_thresholds(trigger_kde, tol=0.95)
# filt = lambda x: (x[:, 0] <= max_t) & (np.sqrt(x[:, 1] ** 2 + x[:, 2] ** 2) <= max_d)
#
# i, j = np.triu_indices(trigger_kde.ndata, k=1)
# chunksize = 10000
# n_iter = i.size / chunksize + 1
# G, I, J = [], [], []
# for n in range(n_iter):
#     print "%u / %u" % (n, n_iter)
#     # get valid pairwise differences
#     this_i = i[n*chunksize:(n+1)*chunksize]
#     I.extend(this_i)
#     this_j = j[n*chunksize:(n+1)*chunksize]
#     J.extend(this_j)
#     d = data[this_j, :] - data[this_i, :]
#     d = d[filt(d)]
#     xd = data[this_j, 1] - data[this_i, 1]
#     yd = data[this_j, 2] - data[this_i, 2]
#     G.extend(trigger_kde.pdf(d[:, 0], d[:, 1], d[:, 2]))


# evaluate trigger KDE at interpoint distances
# start with ALL viable interpoints...
# i, j = np.triu_indices(interpoint.shape[0], k=1)
# pd = pdiff[i, j, :]

# ...and restrict to those within a tolerance to improve efficiency (include 99% of datapoints)
# max_t, max_d = estimation.compute_trigger_thresholds(trigger_kde, tol=0.99)
# pd = pd[(pd[:, 0] <= max_t) & (np.sqrt(pd[:, 1] ** 2 + pd[:, 2] ** 2) <= max_d)]

# G = trigger_kde.pdf(pdiff[i, j, 0], pdiff[i, j, 1], pdiff[i, j, 2])