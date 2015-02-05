__author__ = 'gabriel'
from analysis import cad, validation, hotspot
from point_process import models as pp_models, estimation, validate, plotting as pp_plotting
from database import models
import datetime
from matplotlib import pyplot as plt
import numpy as np
from rpy2 import robjects, rinterface

b_save_to_r = True
output_file = 'kde_variable_bandwidth_nn_assess.Rdata'

# start_date is the FIRST DAY OF THE PREDICTION
start_date = datetime.datetime(2011, 9, 28)
estimate_kwargs = {
    'ct': 1,
    'cd': 0.02
}
model_kwargs = {
    'max_delta_t': 60,
    'max_delta_d': 300,
    'bg_kde_kwargs': {'number_nn': [101, 16], 'strict': False},
    'trigger_kde_kwargs': {'number_nn': 15,
                           'min_bandwidth': [0.5, 30, 30],
                           'strict': False},
    'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, **estimate_kwargs)
}
niter = 50
# niter = 20

poly = cad.get_camden_region()

# load grid and create ROC for use in predictions
qset = models.Division.objects.filter(type='monsuru_250m_grid')
qset = sorted(qset, key=lambda x:int(x.name))
grid_squares = [t.mpoly[0] for t in qset]

num_validation = 100
# num_validation = 10
coverage_20_idx = 81  # this is the closest match to 20pct coverage for the given CAD grid squares
num_sample_points = 50

# end date is the last date retrieved from the database of crimes
# have to cast this to a date since the addition operation automatically produces a datetime
end_date = (start_date + datetime.timedelta(days=num_validation - 1)).date()

kinds = ['burglary', 'shoplifting', 'violence']

sepp_objs = {}
model_objs = {}
res = {}
vb_objs = {}
data_dict = {}
cid_dict = {}

for k in kinds:

    data, t0, cid = cad.get_crimes_from_dump('monsuru_cad_%s' % k)
    # filter: day 210 is 27/9/2011, so use everything LESS THAN 211

    ### SeppValidationFixedModel with centroid ROC sampling

    b_sepp = True
    vb = validate.SeppValidationFixedModel(data=data,
                                           pp_class=pp_models.SeppStochasticNn,
                                           data_index=cid,
                                           spatial_domain=poly,
                                           cutoff_t=211,
                                           model_kwargs=model_kwargs,
                                           )
    vb.set_grid(grid_squares)

    ### SeppValidationFixedModel with integration ROC sampling

    # b_sepp = True
    # vb = validate.SeppValidationFixedModelIntegration(data=data,
    #                                        pp_class=pp_models.SeppStochasticNn,
    #                                        data_index=cid,
    #                                        spatial_domain=poly,
    #                                        cutoff_t=211,
    #                                        model_kwargs=model_kwargs,
    #                                        )
    #
    # vb.set_grid(grid_squares, num_sample_points)
    # res[k] = vb.run(time_step=1, n_iter=num_validation, verbose=True,
    #                 train_kwargs={'niter': niter})
    #
    #
    # sepp_objs[k] = vb.model

    ### SeppValidationPredefinedModel with centroid ROC sampling

    # b_sepp = True
    # training_data = data[data[:, 0] < 211.]
    #
    # # train a model
    # r = pp_models.SeppStochasticNn(data=training_data, **model_kwargs)
    # r.set_seed(42)
    # r.train(niter=niter)
    #
    # sepp_objs[k] = r
    # # disable data sorting (it's already sorted anyway) so that we can lookup cid later
    # vb = validate.SeppValidationPredefinedModel(data=data,
    #                                             model=r,
    #                                             data_index=cid,
    #                                             spatial_domain=poly,
    #                                             cutoff_t=211)
    # vb.set_grid(grid_squares)
    # res[k] = vb.run(time_step=1, n_iter=num_validation, verbose=True)

    ### Historic spatial KDE (Scott bandwidth) with integration sampling
    # time_window = 60
    # b_sepp = False
    # sk = hotspot.SKernelHistoric(60)
    # vb = validation.ValidationIntegration(data,
    #                                       model_class=hotspot.Hotspot,
    #                                       spatial_domain=poly,
    #                                       model_args=(sk,),
    #                                       cutoff_t=211)
    # vb.set_grid(grid_squares, num_sample_points)
    # res[k] = vb.run(time_step=1, n_iter=num_validation, verbose=True,
    #                 train_kwargs={'niter': niter})
    # model_objs[k] = vb.model

    ### Historic spatial KDE (NN bandwidth, 20 NNs) with integration sampling
    # time_window = 60
    # b_sepp = False
    # sk = hotspot.SKernelHistoricVariableBandwidthNn(dt=60, nn=20)
    # vb = validation.ValidationIntegration(data,
    #                                       model_class=hotspot.Hotspot,
    #                                       spatial_domain=poly,
    #                                       model_args=(sk,),
    #                                       cutoff_t=211)
    # vb.set_grid(grid_squares, num_sample_points)

    res[k] = vb.run(time_step=1, n_iter=num_validation, verbose=True,
                    train_kwargs={'niter': niter})
    model_objs[k] = vb.model
    vb_objs[k] = vb
    data_dict[k] = data
    cid_dict[k] = cid


# write data to files
if b_save_to_r:
    var_names = []

    # also store captured crimes array for each crime type, for consistency checking afterwards
    captured_crimes_dict = {}

    for k in kinds:

        if b_sepp:
            main_res = res[k]['full_static']
        else:
            main_res = res[k]

        # crimes captured at 20 pct coverage
        captured_crimes = []
        for i in range(num_validation):
            this_uncap_ids = [xx for xx in main_res['ranked_crime_id'][i][coverage_20_idx:] if xx is not None]
            if len(this_uncap_ids):
                this_uncap_ids = list(np.sort(np.concatenate(this_uncap_ids)))
            else:
                this_uncap_ids = []

            this_cap_ids = [xx for xx in main_res['ranked_crime_id'][i][:coverage_20_idx] if xx is not None]
            if len(this_cap_ids):
                this_cap_ids = list(np.sort(np.concatenate(this_cap_ids)))
            else:
                this_cap_ids = []

            [captured_crimes.append([xx, 1, i + 1]) for xx in this_cap_ids]
            [captured_crimes.append([xx, 0, i + 1]) for xx in this_uncap_ids]

        var_name = 'captured_crimes_20pct_%s' % k
        captured_crimes = np.array(captured_crimes, dtype=int)

        captured_crimes_dict[k] = captured_crimes

        r_vec = robjects.IntVector(captured_crimes.transpose().flatten())  # R has default assignment order 'F' -> transpose
        r_mat = robjects.r['matrix'](r_vec, ncol=3)
        rinterface.globalenv[var_name] = r_mat
        var_names.append(var_name)

        # ranking
        var_name = 'grid_rank_%s' % k
        # need to add 1 to all rankings as Monsuru's IDs are one-indexed and mine are zero-indexed
        # r_vec = robjects.IntVector(main_res['prediction_rank'].flatten() + 1)
        r_vec = robjects.IntVector(np.array(main_res['prediction_rank']).flatten() + 1)
        r_mat = robjects.r['matrix'](r_vec, ncol=num_validation)
        rinterface.globalenv[var_name] = r_mat
        var_names.append(var_name)

        # hit rate by crime count
        var_name = 'crime_count_%s' % k
        # r_vec = robjects.IntVector(main_res['cumulative_crime_count'].flatten())
        r_vec = robjects.IntVector(np.array(main_res['cumulative_crime_count']).flatten())
        r_mat = robjects.r['matrix'](r_vec, ncol=num_validation)
        rinterface.globalenv[var_name] = r_mat
        var_names.append(var_name)

    robjects.r.save(*var_names, file=output_file)

# consistency checks
for k in kinds:

    if b_sepp:
        main_res = res[k]['full_static']
    else:
        main_res = res[k]

    cum_crime = main_res['cumulative_crime']
    cum_crime_count = main_res['cumulative_crime_count']
    crimes_per_day = cum_crime_count[:, -1].astype(float)

    # compute cumul crime fraction from the count
    cum_crime_from_count = (cum_crime_count.transpose() / crimes_per_day).transpose()

    # check equality, ignoring nan
    assert np.all(cum_crime[~np.all(np.isnan(cum_crime), axis=1)] ==
                  cum_crime_from_count[~np.all(np.isnan(cum_crime_from_count), axis=1)]), \
        "Cumulative crime fraction does not match cumulative crime count"

    # compute crimes per day and cumulative crime count from ranked IDs
    crimes_per_day_from_cid = []
    cum_crime_count_from_cid = []
    for i in range(num_validation):
        this_cid = main_res['ranked_crime_id'][i]
        this_cum_crime_count = np.cumsum([len(t) if t is not None else 0 for t in this_cid])
        crimes_per_day_from_cid.append(this_cum_crime_count[-1])
        cum_crime_count_from_cid.append(this_cum_crime_count)

    crimes_per_day_from_cid = np.array(crimes_per_day_from_cid)
    assert np.all(crimes_per_day == crimes_per_day_from_cid), "Crimes per day does not match CIDs"

    cum_crime_count_from_cid = np.array(cum_crime_count_from_cid)
    assert np.all(cum_crime_count == cum_crime_count_from_cid), "Cumulative crime count does not match CIDs"

# plots

for k in kinds:
    pp_plotting.validation_multiplot(res[k])
