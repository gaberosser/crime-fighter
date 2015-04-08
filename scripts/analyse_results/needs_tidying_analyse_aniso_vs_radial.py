__author__ = 'gabriel'
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.collections as collections
import dill

iso_file = '/home/gabriel/pickled_results/chicago_south/max_triggers_grid100_iso_bgfrac5e-1/burglary_120-500-validation.pickle'
ani_file = '/home/gabriel/pickled_results/chicago_south/max_triggers_grid100_ani_bgfrac5e-1/burglary_120-500-validation.pickle'

with open(iso_file, 'r') as f:
    res_iso = dill.load(f)

with open(ani_file, 'r') as f:
    res_ani = dill.load(f)

hr_ani = res_ani['full_static']['cumulative_crime']
hr_iso = res_iso['full_static']['cumulative_crime']

pai_ani = res_ani['full_static']['pai']
pai_iso = res_iso['full_static']['pai']

x_ani = res_ani['full_static']['cumulative_area'].mean(axis=0)
x_iso = res_iso['full_static']['cumulative_area'].mean(axis=0)

# 25th percentile index
idx25_ani = np.where(x_ani > 0.25)[0][0]
idx25_iso = np.where(x_iso > 0.25)[0][0]
idx25 = max(idx25_ani, idx25_iso)

# run wilcox paired rank test
hr_wilcox = np.array([stats.wilcoxon(hr_iso[:, i], hr_ani[:, i], zero_method='wilcox')[1] for i in range(idx25)])
pai_wilcox = np.array([stats.wilcoxon(pai_iso[:, i], pai_ani[:, i], zero_method='wilcox')[1] for i in range(idx25)])


p_crit = 0.05

plt.figure()
plt.plot(x_ani[:idx25_ani], hr_ani.mean(axis=0)[:idx25_ani], label='Anisotropic')
plt.plot(x_iso[:idx25_iso], hr_iso.mean(axis=0)[:idx25_iso], label='Radial')

# just choose one of the cumulative area measures; they differ by < 1%
coll = collections.BrokenBarHCollection.span_where(x_iso[:idx25],
                                                   ymin=0,
                                                   ymax=1.0,
                                                   where=hr_wilcox<0.05,
                                                   facecolor='k',
                                                   edgecolor='none',
                                                   alpha=0.3)
plt.gca().add_collection(coll)
plt.xlim([0, 0.25])
plt.ylim([0, 1])
plt.xlabel('Proportion coverage')
plt.ylabel('Mean hit rate')
plt.legend()

pai_ani_m = pai_ani.mean(axis=0)[:idx25_ani]
pai_iso_m = pai_iso.mean(axis=0)[:idx25_iso]

plt.figure()
plt.plot(x_ani[:idx25_ani], pai_ani_m, label='Anisotropic')
plt.plot(x_iso[:idx25_iso], pai_iso_m, label='Radial')
ymax = np.ceil(10 * max(max(pai_ani_m), max(pai_iso_m))) / 10.
# just choose one of the cumulative area measures; they differ by < 1%
coll = collections.BrokenBarHCollection.span_where(x_iso[:idx25],
                                                   ymin=0,
                                                   ymax=ymax,
                                                   where=pai_wilcox<0.05,
                                                   facecolor='k',
                                                   edgecolor='none',
                                                   alpha=0.3)
plt.gca().add_collection(coll)
plt.xlim([0, 0.25])
plt.ylim([0, ymax])
plt.xlabel('Proportion coverage')
plt.ylabel('Mean PAI')
plt.legend()

plt.show()