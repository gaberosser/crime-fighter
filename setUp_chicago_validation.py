__author__ = 'gabriel'
import numpy as np
import datetime
from analysis import plotting, chicago
import dill
from point_process import estimation, models, validate
from point_process import plotting as pp_plotting
from database.models import ChicagoDivision
from matplotlib import pyplot as plt


# start_date is the FIRST DAY OF THE PREDICTION
start_date = datetime.date(2013, 6, 1)
estimate_kwargs = {
    'ct': 1,
    'cd': 0.02
}
model_kwargs = {
    'max_delta_t': 60,
    'max_delta_d': 300,
    'bg_kde_kwargs': {'number_nn': [101, 16]},
    'trigger_kde_kwargs': {'number_nn': 15,
                           'min_bandwidth': [0.5, 30, 30]},
    'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, **estimate_kwargs)
}
niter = 50

chic = chicago.compute_chicago_region()
southwest = ChicagoDivision.objects.get(name='Southwest').mpoly
south = ChicagoDivision.objects.get(name='South').mpoly
west = ChicagoDivision.objects.get(name='West').mpoly

southwest_buf = southwest.buffer(1500)
south_buf = south.buffer(1500)
west_buf = west.buffer(1500)

training_size = 120
num_validation = 30

# compute number of days in date range
pre_start_date = start_date - datetime.timedelta(days=training_size)
ndays = training_size + num_validation

# end date is the last date retrieved from the database of crimes
# have to cast this to a date since the addition operation automatically produces a datetime
end_date = start_date + datetime.timedelta(days=num_validation - 1)

res, t0 = chicago.get_crimes_by_type(
    crime_type='burglary',
    start_date=pre_start_date,
    end_date=end_date,
)

# train a model
training_data = res[res[:, 0] < training_size]
r = models.SeppStochasticNn(data=training_data, **model_kwargs)
r.train(niter=niter)

# centroid method
vb_centroid = {}
res_centroid = {}

vb_centroid_500 = validate.SeppValidationPredefinedModel(data=res,
                                            model=r,
                                            spatial_domain=south,
                                            cutoff_t=training_size)
vb_centroid_500.set_grid(500)
res_centroid[500] = vb_centroid_500.run(time_step=1, n_iter=num_validation, verbose=True)
vb_centroid[500] = vb_centroid_500

vb_centroid_250 = validate.SeppValidationPredefinedModel(data=res,
                                            model=r,
                                            spatial_domain=south,
                                            cutoff_t=training_size)
vb_centroid_250.set_grid(250)
res_centroid[250] = vb_centroid_250.run(time_step=1, n_iter=num_validation, verbose=True)
vb_centroid[250] = vb_centroid_250

params = [
    (500, 10),
    (500, 50),
    (250, 10),
    (250, 50)
]

vb_sample_points = {}
res_sample_points = {}

for t in params:
    vb_sample_points[t] = validate.SeppValidationPredefinedModelIntegration(data=res,
                                                           model=r,
                                                           spatial_domain=south,
                                                           cutoff_t=training_size)
    vb_sample_points[t].set_grid(*t)
    res_sample_points[t] = vb_sample_points[t].run(time_step=1, n_iter=num_validation, verbose=True)

# now grab the next 30 days of data for training and train a new model
# we're looking to see how much it changes (if at all)
training_size += num_validation
start_date = end_date + datetime.timedelta(days=1)
pre_start_date = start_date - datetime.timedelta(days=training_size)
end_date = start_date + datetime.timedelta(days=num_validation - 1)

res, t0 = chicago.get_crimes_by_type(
    crime_type='burglary',
    start_date=pre_start_date,
    end_date=end_date,
)

training_data = res[res[:, 0] < training_size]
r2 = models.SeppStochasticNn(data=training_data, **model_kwargs)
r2.train(niter=niter)


## plotting
centroids500 = vb_centroid[500].roc.sample_points
centroids250 = vb_centroid[250].roc.sample_points
sp500_10 = vb_sample_points[(500, 10)].roc.sample_points
sp500_50 = vb_sample_points[(500, 50)].roc.sample_points

fig_size = (12.6,  14.5)
zoom_fig_size = (9.81333333,  9.66666667)
full_bounding = (444650, 4622000, 455250, 4634150)
zoom_bounding = (450180.36053909821, 4622427.5857217703, 454025.44137770293, 4626215.1993195796)


def zoom_in():
    ax = plt.gca()
    fig = plt.gcf()
    fig.set_size_inches(zoom_fig_size)
    ax.set_position([0, 0, 1, 1])
    ax.axis('tight')
    ax.set_xlim([zoom_bounding[0], zoom_bounding[2]])
    ax.set_ylim([zoom_bounding[1], zoom_bounding[3]])


def zoom_out():
    ax = plt.gca()
    fig = plt.gcf()
    fig.set_size_inches(fig_size)
    ax.set_position([0, 0, 1, 1])
    ax.axis('tight')
    ax.set_xlim([full_bounding[0], full_bounding[2]])
    ax.set_ylim([full_bounding[1], full_bounding[3]])

# this should be the ONLY time we compute the values of the risk surface density
xs, ys, zs = pp_plotting.prediction_heatmap(r, 121, kind='static', poly=south, dx=30)

plotting.plot_surface_on_polygon((xs, ys, zs), poly=south, fmax=0.99)

zoom_out()
plt.savefig('full_static_t121.png')
plotting.plot_geodjango_shapes(vb_centroid[500].roc.igrid, facecolor='none', ax=plt.gca(), alpha=0.3, lw=2, set_axes=False)
plt.savefig('full_static_t121_grid500.png')

pts = plt.plot(centroids500.getdim(0), centroids500.getdim(1), 'o', color='k', markerfacecolor='w', alpha=0.4, lw=1.5)
plt.savefig('full_static_t121_grid500_centroids.png')
zoom_in()
plt.savefig('full_static_t121_grid500_centroids_zoom.png')
zoom_out()
[x.set_visible('off') for x in pts]

pts = plt.plot(sp500_10.getdim(0), sp500_10.getdim(1), 'o', color='k', markerfacecolor='w', alpha=0.4, lw=1.5)
plt.savefig('full_static_t121_grid500_spoint10.png')
zoom_in()
plt.savefig('full_static_t121_grid500_spoint10_zoom.png')
zoom_out()
[x.set_visible('off') for x in pts]

pts = plt.plot(sp500_50.getdim(0), sp500_50.getdim(1), 'o', color='k', markerfacecolor='w', alpha=0.4, lw=1.5)
plt.savefig('full_static_t121_grid500_spoint50.png')
zoom_in()
plt.savefig('full_static_t121_grid500_spoint50_zoom.png')
zoom_out()
[x.set_visible('off') for x in pts]

# show order of grid square selection
idx_centroid500 = res_centroid[500]['full_static']['prediction_rank'][0]
idx_sp500_10 = res_sample_points[(500, 10)]['full_static']['prediction_rank'][0]
idx_sp500_50 = res_sample_points[(500, 50)]['full_static']['prediction_rank'][0]

plt.close('all')
plotting.plot_surface_on_polygon((xs, ys, zs), poly=south, fmax=0.99)
plotting.plot_geodjango_shapes(vb_centroid[500].roc.igrid, facecolor='none', ax=plt.gca(), alpha=0.3, lw=2, set_axes=False)
zoom_out()
for n in range(75):
    plotting.plot_geodjango_shapes(vb_centroid[500].roc.igrid[idx_centroid500[n]], facecolor='w', hatch='\\', edgecolor='none', alpha=0.6, set_axes=False)
    plt.savefig('full_static_t121_grid500_centroid_%02dpred.png' % (n+1))

plt.close('all')
plotting.plot_surface_on_polygon((xs, ys, zs), poly=south, fmax=0.99)
plotting.plot_geodjango_shapes(vb_centroid[500].roc.igrid, facecolor='none', ax=plt.gca(), alpha=0.3, lw=2, set_axes=False)
zoom_out()
for n in range(75):
    plotting.plot_geodjango_shapes(vb_sample_points[(500, 10)].roc.igrid[idx_sp500_10[n]], facecolor='w', hatch='\\', edgecolor='none', alpha=0.6, set_axes=False)
    plt.savefig('full_static_t121_grid500_spoint10_%02dpred.png' % (n+1))

plt.close('all')
plotting.plot_surface_on_polygon((xs, ys, zs), poly=south, fmax=0.99)
plotting.plot_geodjango_shapes(vb_centroid[500].roc.igrid, facecolor='none', ax=plt.gca(), alpha=0.3, lw=2, set_axes=False)
zoom_out()
for n in range(75):
    plotting.plot_geodjango_shapes(vb_sample_points[(500, 50)].roc.igrid[idx_sp500_50[n]], facecolor='w', hatch='\\', edgecolor='none', alpha=0.6, set_axes=False)
    plt.savefig('full_static_t121_grid500_spoint50_%02dpred.png' % (n+1))
