__author__ = 'gabriel'
import numpy as np
import pickle
from matplotlib import pyplot as plt

labels = ['SEPP 1dayg100','SEPP 2dayg100', 'SEPP 1dayg20', 'SEPP 2dayg20', 'NNKDE 1dayg100', 'NNKDE 2dayg100', 'NNKDE 1dayg20', 'NNKDE 2dayg20']

f = open('/home/gabriel/Documents/cad_validation_nicl1.pickle', 'r')
nicl1 = [pickle.load(f)[0] for i in range(8)]

f = open('/home/gabriel/Documents/cad_validation_nicl3.pickle', 'r')
nicl3 = [pickle.load(f)[0] for i in range(8)]

f = open('/home/gabriel/Documents/cad_validation_nicl67.pickle', 'r')
nicl67 = [pickle.load(f)[0] for i in range(8)]

x1 = []
mu1 = []
std1 = []

for res in nicl1:
    if 'cumulative_crime' in res:
        x1.append(np.linspace(0, 1, len(res['cumulative_crime'][0])))
        mu1.append(np.nanmean(res['cumulative_crime'], axis=0))
        std1.append(np.nanstd(res['cumulative_crime'], axis=0))
    else:
        x1.append(np.linspace(0, 1, len(res['cumulative_crime_full'][0])))
        mu1.append(np.nanmean(res['cumulative_crime_full'], axis=0))
        std1.append(np.nanstd(res['cumulative_crime_full'], axis=0))


x3 = []
mu3 = []
std3 = []

for res in nicl3:
    if 'cumulative_crime' in res:
        x3.append(np.linspace(0, 1, len(res['cumulative_crime'][0])))
        mu3.append(np.nanmean(res['cumulative_crime'], axis=0))
        std3.append(np.nanstd(res['cumulative_crime'], axis=0))
    else:
        x3.append(np.linspace(0, 1, len(res['cumulative_crime_full'][0])))
        mu3.append(np.nanmean(res['cumulative_crime_full'], axis=0))
        std3.append(np.nanstd(res['cumulative_crime_full'], axis=0))


x67 = []
mu67 = []
std67 = []

for res in nicl67:
    if 'cumulative_crime' in res:
        x67.append(np.linspace(0, 1, len(res['cumulative_crime'][0])))
        mu67.append(np.nanmean(res['cumulative_crime'], axis=0))
        std67.append(np.nanstd(res['cumulative_crime'], axis=0))
    else:
        x67.append(np.linspace(0, 1, len(res['cumulative_crime_full'][0])))
        mu67.append(np.nanmean(res['cumulative_crime_full'], axis=0))
        std67.append(np.nanstd(res['cumulative_crime_full'], axis=0))

styles = ['k-', 'k--', 'r-', 'r--', 'g-', 'g--', 'b-', 'b--']

fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(131)
ax3 = fig.add_subplot(132, sharex=ax1, sharey=ax1)
ax67 = fig.add_subplot(133, sharex=ax1, sharey=ax1)

[ax1.plot(a, b, c) for a, b, c in zip(x1, mu1, styles)]
ax1.set_title('Violence Against The Person')

[ax3.plot(a, b, c) for a, b, c in zip(x3, mu3, styles)]
ax3.set_title('Burglary From Dwelling')

[ax67.plot(a, b, c) for a, b, c in zip(x67, mu67, styles)]
ax67.legend(labels, loc=4)
ax67.set_title('Theft Of/From Vehicle')

ax3.yaxis.set_visible(False)
ax67.yaxis.set_visible(False)

ax3.set_xlabel('Fraction area')
ax1.set_ylabel('Fraction crime')

plt.tight_layout()

##################

styles = ['k--', 'k-', 'r--', 'r-']
labels = ['SEPP grid 100', 'SEPP grid 20', 'NNKDE grid 100', 'NNKDE grid 20']

c = 0.2
xlim = [0, c]

fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(131, xlim=xlim)
ax3 = fig.add_subplot(132, sharex=ax1, sharey=ax1)
ax67 = fig.add_subplot(133, sharex=ax1, sharey=ax1)

[ax1.plot(x1[i], mu1[i], styles[i/2]) for i in range(8) if i % 2 == 0]
ax1.set_title('Violence Against The Person')

[ax3.plot(x3[i], mu3[i], styles[i/2]) for i in range(8) if i % 2 == 0]
ax3.set_title('Burglary From Dwelling')

[ax67.plot(x67[i], mu67[i], styles[i/2]) for i in range(8) if i % 2 == 0]
ax67.set_title('Theft Of/From Vehicle')

ax3.yaxis.set_visible(False)
ax67.yaxis.set_visible(False)

ax3.set_xlabel('Fraction area', fontsize=20)
ax1.set_ylabel('Fraction crime', fontsize=20)

plt.setp(ax1.get_xticklabels(), fontsize=16)
plt.setp(ax3.get_xticklabels(), fontsize=16)
plt.setp(ax67.get_xticklabels(), fontsize=16)
plt.setp(ax1.get_yticklabels(), fontsize=16)

ax1.set_xlim(xlim)

ymax1 = max([np.max(b[a < c]) for a, b in zip(x1, mu1)])
ymax3 = max([np.max(b[a < c]) for a, b in zip(x3, mu3)])
ymax67 = max([np.max(b[a < c]) for a, b in zip(x67, mu67)])

ymax = max(ymax1, ymax3, ymax67)

ax1.set_ylim([0., 1.02 * ymax])

ax3.legend(labels, loc=2)

plt.tight_layout()

##################

labels = ['SEPP', 'NNKDE']

# 20 x 20 grid
# idxsepp = 2
# idxnnkde = 6

# 100 x 100 grid
idxsepp = 0
idxnnkde = 4

c = 0.2
xlim = [0, c]

fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(131, xlim=xlim)
ax3 = fig.add_subplot(132, sharex=ax1, sharey=ax1)
ax67 = fig.add_subplot(133, sharex=ax1, sharey=ax1)

ax1.plot(x1[idxsepp], mu1[idxsepp], 'k-')
ax1.plot(x1[idxnnkde], mu1[idxnnkde], 'r-')
ax1.plot([0, c], [0, c], 'k--')
ax1.set_title('Violence Against The Person')

ax3.plot(x3[idxsepp], mu3[idxsepp], 'k-')
ax3.plot(x3[idxnnkde], mu3[idxnnkde], 'r-')
ax3.plot([0, c], [0, c], 'k--')
ax3.set_title('Burglary From Dwelling')

ax67.plot(x67[idxsepp], mu67[idxsepp], 'k-')
ax67.plot(x67[idxnnkde], mu67[idxnnkde], 'r-')
ax67.plot([0, c], [0, c], 'k--')
ax67.set_title('Theft Of/From Vehicle')

ax3.yaxis.set_visible(False)
ax67.yaxis.set_visible(False)

ax3.set_xlabel('Fraction area', fontsize=20)
ax1.set_ylabel('Fraction crime', fontsize=20)

plt.setp(ax1.get_xticklabels(), fontsize=16)
plt.setp(ax3.get_xticklabels(), fontsize=16)
plt.setp(ax67.get_xticklabels(), fontsize=16)
plt.setp(ax1.get_yticklabels(), fontsize=16)

ax1.set_xlim(xlim)

ymax1 = max([np.max(b[a < c]) for a, b in zip(x1, mu1)])
ymax3 = max([np.max(b[a < c]) for a, b in zip(x3, mu3)])
ymax67 = max([np.max(b[a < c]) for a, b in zip(x67, mu67)])

ymax = max(ymax1, ymax3, ymax67)

ax1.set_ylim([0., 1.02 * ymax])

ax3.legend(labels, loc=2)

plt.tight_layout()