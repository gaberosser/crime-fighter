__author__ = 'gabriel'
import datetime
from analysis import cad, spatial, chicago
from point_process import models as pp_models, estimation, validate, plots as pp_plotting
from database import models
from validation import hotspot, validation
import numpy as np
from rpy2 import robjects, rinterface
import csv


def get_chicago_polys():
    sides = models.ChicagoDivision.objects.filter(type='chicago_side')
    res = {}
    for s in sides:
        res[s.name] = spatial.geodjango_to_shapely(s.mpoly.simplify())
    return res


def get_chicago_grid(poly):
    ipoly, fpoly, full = spatial.create_spatial_grid(poly, 250)
    return fpoly


def create_chicago_grid_squares_shapefile(outfile):
    fpoly = get_chicago_grid()
    field_description = {'id': {'fieldType': 'N'}}
    id = range(1, len(fpoly) + 1)
    spatial.write_polygons_to_shapefile(outfile,
                                        fpoly,
                                        field_description=field_description,
                                        id=id)


def get_chicago_data(primary_types=None):
    start_date = datetime.date(2011, 3, 1)
    end_date = datetime.date(2012, 1, 6)
    domain = models.ChicagoDivision.objects.get(name='South').mpoly.simplify()
    if primary_types is None:
        primary_types = (
            'burglary',
            'assault',
            'motor vehicle theft'
        )

    data = {}
    for pt in primary_types:
        data[pt] = chicago.get_crimes_by_type(pt, start_date=start_date,
                                              end_date=end_date,
                                              domain=domain,
                                              convert_dates=True)
    return data


def create_chicago_data_csv_file(filestem='chicago_data'):
    data = get_chicago_data()
    for pt, (res, t0, cid) in data.items():
        monsuru_weird_dates = res[:, 0].astype(int) + 40603  # 40603 corresponds to 1/3/2011
        x = res[:, 1]
        y = res[:, 2]
        data = [{'id': a, 'date': b, 'x':c, 'y': d} for a, b, c, d in zip(cid, monsuru_weird_dates, x, y)]
        filename = '%s_%s.csv' % (filestem, pt.replace(' ', '_'))
        with open(filename, 'w') as f:
            c = csv.DictWriter(f, fieldnames=('id', 'date', 'x', 'y'))
            c.writeheader()
            c.writerows(data)


def apply_historic_kde(data,
                       data_index,
                       domain,
                       grid_squares=None,
                       num_sample_points=10,
                       time_window=60):
    ### Historic spatial KDE (Scott bandwidth) with integration sampling
    sk = hotspot.SKernelHistoric(time_window)
    vb = validation.ValidationIntegration(data,
                                          model=sk,
                                          data_index=data_index,
                                          spatial_domain=domain,
                                          cutoff_t=211)
    if grid_squares:
        vb.roc.set_sample_units_predefined(grid_squares, num_sample_points)
    else:
        vb.set_sample_units(250, num_sample_points)
    res = vb.run(time_step=1, n_iter=100, verbose=True)
    return res


def apply_historic_kde_variable_bandwidth(data,
                                          data_index,
                                          domain,
                                          grid_squares=None,
                                          num_nn=20,
                                          num_sample_points=10,
                                          time_window=60):
    sk = hotspot.SKernelHistoricVariableBandwidthNn(dt=time_window, nn=num_nn)
    vb = validation.ValidationIntegration(data,
                                          model=sk,
                                          data_index=data_index,
                                          spatial_domain=domain,
                                          cutoff_t=211)
    if grid_squares:
        vb.roc.set_sample_units_predefined(grid_squares, num_sample_points)
    else:
        vb.set_sample_units(250, num_sample_points)
    res = vb.run(time_step=1, n_iter=100, verbose=True)
    return res


def apply_sepp_stochastic_nn(data,
                             data_index,
                             domain,
                             grid_squares=None,
                             max_t=60,
                             max_d=500,
                             niter_training=50,
                             num_sample_points=10,
                             seed=43):

    sepp = pp_models.SeppStochasticNn(max_delta_t=max_t,
                                      max_delta_d=max_d,
                                      estimation_function=lambda x, y: estimation.estimator_bowers(x, y, ct=1, cd=10),
                                      seed=seed)
    vb = validate.SeppValidationFixedModelIntegration(data=data,
                                                      model=sepp,
                                                      data_index=data_index,
                                                      spatial_domain=domain,
                                                      cutoff_t=211)

    if grid_squares:
        vb.roc.set_sample_units_predefined(grid_squares, num_sample_points)
    else:
        vb.set_sample_units(250, num_sample_points)
    res = vb.run(time_step=1, n_iter=100, verbose=True,
                 train_kwargs={'niter': niter_training})
    return res


def dump_results_to_rdata_file(output_file, b_sepp=False, **components):
    """
    Save a collection of results to an Rdata file.
    :param filename: The full path to the output file
    :param b_sepp: Boolean. If True, the results are from an SEPP object.
    :param components: Dictionary containing the components to include in the output. The key gives the crime type,
    the value is the validation results dictionary.
    :return:
    """
    var_names = []
    kinds = components.keys()
    captured_crimes_dict = {}

    for k in kinds:

        if b_sepp:
            main_res = components[k]['full_static']
        else:
            main_res = components[k]

        num_validation = main_res['cumulative_area'].shape[0]
        num_sample_units = main_res['cumulative_area'].shape[1]
        coverage_20_idx = round(num_sample_units * 0.2) - 1

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
        r_vec = robjects.IntVector(np.array(main_res['prediction_rank']).flatten() + 1)
        r_mat = robjects.r['matrix'](r_vec, ncol=num_validation)
        rinterface.globalenv[var_name] = r_mat
        var_names.append(var_name)

        # hit rate by crime count
        var_name = 'crime_count_%s' % k
        r_vec = robjects.IntVector(np.array(main_res['cumulative_crime_count']).flatten())
        r_mat = robjects.r['matrix'](r_vec, ncol=num_validation)
        rinterface.globalenv[var_name] = r_mat
        var_names.append(var_name)

    robjects.r.save(*var_names, file=output_file)
    return captured_crimes_dict

if __name__ == '__main__':

    res_sepp = {}
    res_kde = {}
    vb_sepp = {}
    vb_kde = {}

    # chicago South side validation
    south = get_chicago_polys()['South']
    grid_squares = get_chicago_grid(south)
    data = get_chicago_data()

    for k in data.keys():
        this_data = data[k][0]
        this_data_index = data[k][-1]
        this_res_kde = apply_historic_kde_variable_bandwidth(this_data,
                                                             this_data_index,
                                                             south,
                                                             grid_squares=grid_squares)
        this_res_sepp = apply_sepp_stochastic_nn(this_data,
                                                 this_data_index,
                                                 south,
                                                 grid_squares=grid_squares,
                                                 max_t=90)
        res_kde[k] = this_res_kde
        res_sepp[k] = this_res_sepp

    dump_results_to_rdata_file('chicago_south_side_variable_bandwidth_kde_nn.Rdata', **res_kde)
    dump_results_to_rdata_file('chicago_south_side_sepp_stochastic_nn.Rdata', b_sepp=True, **res_sepp)

    # b_save_to_r = True
    # output_file = 'kde_variable_bandwidth_nn_assess.Rdata'
    #
    # # start_date is the FIRST DAY OF THE PREDICTION
    # start_date = datetime.datetime(2011, 9, 28)
    # estimate_kwargs = {
    #     'ct': 1,
    #     'cd': 0.02
    # }
    # model_kwargs = {
    #     'max_delta_t': 60,
    #     'max_delta_d': 300,
    #     'bg_kde_kwargs': {'number_nn': [101, 16], 'strict': False},
    #     'trigger_kde_kwargs': {'number_nn': 15,
    #                            'min_bandwidth': [0.5, 30, 30],
    #                            'strict': False},
    #     'estimation_function': lambda x, y: estimation.estimator_bowers(x, y, **estimate_kwargs)
    # }
    # niter = 50
    # # niter = 20
    #
    # poly = cad.get_camden_region()
    #
    # # load grid and create ROC for use in predictions
    # qset = models.Division.objects.filter(type='monsuru_250m_grid')
    # qset = sorted(qset, key=lambda x:int(x.name))
    # grid_squares = [t.mpoly[0] for t in qset]

    # num_validation = 100
    # coverage_20_idx = 81  # this is the closest match to 20pct coverage for the given CAD grid squares
    # num_sample_points = 50

    # end date is the last date retrieved from the database of crimes
    # have to cast this to a date since the addition operation automatically produces a datetime
    # end_date = (start_date + datetime.timedelta(days=num_validation - 1)).date()

    # kinds = ['burglary', 'shoplifting', 'violence']

    # sepp_objs = {}
    # model_objs = {}
    # res = {}
    # vb_objs = {}
    # data_dict = {}
    # cid_dict = {}

    # for k in kinds:

        # data, t0, cid = cad.get_crimes_from_dump('monsuru_cad_%s' % k)
        # filter: day 210 is 27/9/2011, so use everything LESS THAN 211

        ### SeppValidationFixedModel with centroid ROC sampling

        # b_sepp = True
        # vb = validate.SeppValidationFixedModel(data=data,
        #                                        pp_class=pp_models.SeppStochasticNn,
        #                                        data_index=cid,
        #                                        spatial_domain=poly,
        #                                        cutoff_t=211,
        #                                        model_kwargs=model_kwargs,
        #                                        )
        # vb.set_sample_units(grid_squares)

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
        # vb.set_sample_units(grid_squares, num_sample_points)
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
        # vb.set_sample_units(grid_squares)
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
        # vb.set_sample_units(grid_squares, num_sample_points)
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
        # vb.set_sample_units(grid_squares, num_sample_points)
        # res[k] = vb.run(time_step=1, n_iter=num_validation, verbose=True,
        #                 train_kwargs={'niter': niter})
        # model_objs[k] = vb.model
        # vb_objs[k] = vb
        # data_dict[k] = data
        # cid_dict[k] = cid
