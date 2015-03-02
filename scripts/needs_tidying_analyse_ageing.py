__author__ = 'gabriel'
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy import stats
from scripts import OUT_DIR

f = open('coverage20.pickle', 'r') ## TODO: move this to the appropriate location
res = pickle.load(f)

pai30 = {}
mean_pai30 = {}

for name in res:
    t = res[name]
    this_pai = {}
    mean_pai = {}
    for k in t:
        mean_pai[k] = [np.nanmean(t[k]['pai'][i:i+30]) for i in range(0, len(t[k]['pai']), 30)]
        this_pai[k] = [t[k]['pai'][i:i+30] for i in range(0, len(t[k]['pai']), 30)]
    mean_pai30[name] = mean_pai
    pai30[name] = this_pai

    base_pai = mean_pai[277]
    refreshed_pai = [mean_pai[k][0] for k in sorted(mean_pai.keys())]

    # fig = plt.figure(name)
    # ax = fig.add_subplot(111)
    # ax.plot(base_pai)
    # ax.plot(refreshed_pai)
    # ax.set_ylim([0, max(np.max(base_pai), np.max(refreshed_pai)) * 1.05])

    pairwise = {}
    t0s = sorted(t.keys())
    n = len(t0s)
    for t in range(1, n):  # time lags
        data = []
        for j in range(t, n):  # rolling window
            new = this_pai[t0s[j]][0]
            old = this_pai[t0s[j - t]][t]
            data.extend(list((new - old)[~np.isnan(new - old)]))
        pairwise[t] = np.array(data)

    x = np.arange(1, n)
    ym = np.array([pairwise[t].mean() for t in range(1, n)])
    y25 = np.array([sorted(pairwise[t])[int(0.25 * len(pairwise[t]))] for t in range(1, n)])
    y75 = np.array([sorted(pairwise[t])[int(0.75 * len(pairwise[t]))] for t in range(1, n)])
    y5 = np.array([sorted(pairwise[t])[int(0.05 * len(pairwise[t]))] for t in range(1, n)])
    y95 = np.array([sorted(pairwise[t])[int(0.95 * len(pairwise[t]))] for t in range(1, n)])
    skews = np.array([stats.skew(pairwise[t]) for t in range(1, n)])
    np_skews = np.array([(pairwise[t].mean() - np.median(pairwise[t])) / pairwise[t].std() for t in range(1, n)])

    fig = plt.figure(name)
    ax = fig.add_subplot(111)
    ax.fill_between(x, y25, y75, facecolor='r', edgecolor='none', alpha=0.3)
    ax.fill_between(x, y5, y95, facecolor='r', edgecolor='none', alpha=0.3)
    ax.plot(x, ym, 'k')

    plt.figure(name + '_skew')
    plt.plot(x, skews)
    plt.plot(x, np_skews)