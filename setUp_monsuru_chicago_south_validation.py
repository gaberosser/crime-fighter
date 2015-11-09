__author__ = 'gabriel'
import datetime
from analysis import cad, spatial, chicago
from point_process import models as pp_models, estimation, validate, plots as pp_plotting
from database import models
from validation import hotspot, validation
import numpy as np
from rpy2 import robjects, rinterface
import csv
import os
import settings

T0 = 40603
INITIAL_CUTOFF = 212
DATA_CSV_DIR = os.path.join(settings.DATA_DIR, 'chicago', 'monsuru_data')

def get_chicago_polys(as_shapely=True):
    sides = models.ChicagoDivision.objects.filter(type='chicago_side')
    res = {}
    for s in sides:
        if as_shapely:
            res[s.name] = spatial.geodjango_to_shapely(s.mpoly.simplify())
        else:
            res[s.name] = s.mpoly.simplify()
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


def get_chicago_data(primary_types=None, domain=None):
    start_date = datetime.date(2011, 3, 1)
    end_date = datetime.date(2012, 1, 6)
    if domain is None:
        domain = models.ChicagoDivision.objects.get(name='South').mpoly.simplify()
    if primary_types is None:
        primary_types = (
            'burglary',
            'assault',
            'motor vehicle theft'
        )

    data = {}
    for pt in primary_types:
        key = pt.replace(' ', '_')
        data[key] = chicago.get_crimes_by_type(pt, start_date=start_date,
                                               end_date=end_date,
                                               domain=domain,
                                               convert_dates=True)
    return data


def load_chicago_data_from_csv(indir=DATA_CSV_DIR):
    filenames = {
        'burglary': 'burglary_Crimes_SS.csv',
        'assault': 'assault_Crimes_SS.csv',
        'motor_vehicle_theft': 'motorVeh_Crimes_SS.csv',
    }
    data = {}
    for k, v in filenames.items():
        with open(os.path.join(indir, v), 'r') as f:
            c = csv.DictReader(f)
            this_cid = []
            this_data = []
            for row in c:
                this_cid.append(int(row['SN']))
                this_data.append([
                    float(row['T']), float(row['X']), float(row['Y'])
                ])
            this_cid = np.array(this_cid)
            this_data = np.array(this_data)
            data[k] = this_data, min(this_data[:, 0]), this_cid
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
                                          cutoff_t=INITIAL_CUTOFF + T0)
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
                                          cutoff_t=INITIAL_CUTOFF + T0)

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
                             max_t=90,
                             max_d=500,
                             niter_training=50,
                             num_sample_points=10,
                             seed=43):

    est_fun = lambda x, y: estimation.estimator_bowers_fixed_proportion_bg(x, y, ct=1, cd=10, frac_bg=0.5)
    trigger_kde_kwargs = {'strict': False}
    bg_kde_kwargs = dict(trigger_kde_kwargs)

    sepp = pp_models.SeppStochasticNn(data=data,
                                      max_delta_t=max_t,
                                      max_delta_d=max_d,
                                      seed=seed,
                                      estimation_function=est_fun,
                                      trigger_kde_kwargs=trigger_kde_kwargs,
                                      bg_kde_kwargs=bg_kde_kwargs)

    vb = validate.SeppValidationFixedModelIntegration(data=data,
                                                      model=sepp,
                                                      data_index=data_index,
                                                      spatial_domain=domain,
                                                      cutoff_t=INITIAL_CUTOFF + T0)

    if grid_squares:
        vb.roc.set_sample_units_predefined(grid_squares, num_sample_points)
    else:
        vb.set_sample_units(250, num_sample_points)
    res = vb.run(time_step=1, n_iter=100, verbose=True,
                 train_kwargs={'niter': niter_training})
    return res


def dump_results_to_rdata_file(output_file, b_sepp=False, suffix='', **components):
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

        var_name = 'captured_crimes_20pct_%s%s' % (k, suffix)
        captured_crimes = np.array(captured_crimes, dtype=int)

        captured_crimes_dict[k] = captured_crimes

        r_vec = robjects.IntVector(captured_crimes.transpose().flatten())  # R has default assignment order 'F' -> transpose
        r_mat = robjects.r['matrix'](r_vec, ncol=3)
        rinterface.globalenv[var_name] = r_mat
        var_names.append(var_name)

        # ranking
        var_name = 'grid_rank_%s%s' % (k, suffix)
        # need to add 1 to all rankings as Monsuru's IDs are one-indexed and mine are zero-indexed
        r_vec = robjects.IntVector(np.array(main_res['prediction_rank']).flatten() + 1)
        r_mat = robjects.r['matrix'](r_vec, ncol=num_validation)
        rinterface.globalenv[var_name] = r_mat
        var_names.append(var_name)

        # hit rate by crime count
        var_name = 'crime_count_%s%s' % (k, suffix)
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
    domain = get_chicago_polys()['South']
    # domain = get_chicago_polys()['North']
    grid_squares = get_chicago_grid(domain)
    # data = get_chicago_data(domain=domain)
    data = load_chicago_data_from_csv()

    for k in data.keys():
    # for k in ['assault', ]:
        this_data = data[k][0]
        this_data_index = data[k][-1]
        try:
            this_res_kde = apply_historic_kde_variable_bandwidth(this_data,
                                                                 this_data_index,
                                                                 domain,
                                                                 grid_squares=grid_squares)
            res_kde[k] = this_res_kde
        except Exception as exc:
            print repr(exc)

        try:
            this_res_sepp = apply_sepp_stochastic_nn(this_data,
                                                     this_data_index,
                                                     domain,
                                                     grid_squares=grid_squares,
                                                     max_t=90,
                                                     max_d=500)
            res_sepp[k] = this_res_sepp
        except Exception as exc:
            print repr(exc)

    dump_results_to_rdata_file('chicago_south_side_variable_bandwidth_kde_nn.Rdata',
                               suffix='_ss', **res_kde)
    dump_results_to_rdata_file('chicago_south_side_sepp_stochastic_nn.Rdata', b_sepp=True,
                               suffix='_ss', **res_sepp)
