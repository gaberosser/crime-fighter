__author__ = 'gabriel'
from analysis import cad, chicago
from point_process import models as pp_models, estimation, validate
import numpy as np
from validation import validation, hotspot
import datetime
import dill
from scripts import OUT_DIR
import os


GRID_LENGTH = 250
N_PER_GRID = 50

def run_chicago_validation(poly, poly_name):
    cutoff = 212
    outdir = os.path.join(OUT_DIR, 'validation', 'chicago', poly_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    start_date = datetime.date(2011, 3, 1)
    end_date = datetime.date(2012, 3, 31)
    data, t0, cid = chicago.get_crimes_by_type(start_date=start_date,
                                               end_date=end_date,
                                               domain=poly)

    trigger_kde_kwargs = {'strict': False}
    bg_kde_kwargs = dict(trigger_kde_kwargs)

    kde_s = hotspot.SKernelHistoricVariableBandwidthNn(60, nn=20)

    vb_s = validation.ValidationIntegration(data,
                                            kde_s,
                                            data_index=cid,
                                            spatial_domain=poly,
                                            cutoff_t=cutoff)
    vb_s.set_sample_units(GRID_LENGTH, n_sample_per_grid=N_PER_GRID)
    res_s = vb_s.run(time_step=1, n_iter=100)
    name = 'skde_nn'
    with open(os.path.join(outdir, name), 'w') as f:
        dill.dump(res_s, f)

    kde_st = hotspot.STLinearSpaceExponentialTime(400., 10.)  # 400m, 10ln2 days half life
    vb_st = validation.ValidationIntegration(data,
                                            kde_st,
                                            data_index=cid,
                                            spatial_domain=poly,
                                            cutoff_t=cutoff)
    vb_st.set_sample_units(GRID_LENGTH, n_sample_per_grid=N_PER_GRID)
    res_st = vb_st.run(time_step=1, n_iter=100)
    name = 'stkde_nn'
    with open(os.path.join(outdir, name), 'w') as f:
        dill.dump(res_st, f)

    try:
        sepp = pp_models.SeppStochasticNn(data=data,
                                          max_delta_t=150,
                                          max_delta_d=500,
                                          seed=42,
                                          estimation_function=lambda x, y: estimation.estimator_exp_gaussian(x, y, ct=.1, cd=50, frac_bg=None),
                                          trigger_kde_kwargs=trigger_kde_kwargs,
                                          bg_kde_kwargs=bg_kde_kwargs,
                                          remove_coincident_pairs=False)

        vb_sepp = validate.SeppValidationFixedModel(data,
                                                    sepp,
                                                    data_index=cid,
                                                    spatial_domain=poly,
                                                    cutoff_t=cutoff)
        vb_sepp.set_sample_units(vb_s.roc)
        res_sepp = vb_sepp.run(time_step=1, n_iter=100, train_kwargs={'niter': 200})
    except Exception:
        res_sepp = None
    name = 'sepp'
    with open(os.path.join(outdir, name), 'w') as f:
        dill.dump(res_sepp, f)

    try:
        sepp_refl = pp_models.SeppStochasticNnReflected(
            data=data,
            max_delta_t=150,
            max_delta_d=500,
            seed=42,
            estimation_function=lambda x, y: estimation.estimator_exp_gaussian(x, y, ct=.1, cd=50, frac_bg=None),
            trigger_kde_kwargs=trigger_kde_kwargs,
            bg_kde_kwargs=bg_kde_kwargs,
            remove_coincident_pairs=False
        )
        vb_sepp_refl = validate.SeppValidationFixedModel(data,
                                                    sepp_refl,
                                                    data_index=cid,
                                                    spatial_domain=poly,
                                                    cutoff_t=cutoff)
        vb_sepp_refl.set_sample_units(vb_s.roc)
        res_sepp_refl = vb_sepp_refl.run(time_step=1, n_iter=100, train_kwargs={'niter': 200})
    except Exception:
        res_sepp_refl = None
    name = 'sepp_refl'
    with open(os.path.join(outdir, name), 'w') as f:
        dill.dump(res_sepp_refl, f)

    try:
        sepp_iso = pp_models.SeppStochasticNnIsotropicTrigger(
            data=data,
            max_delta_t=150,
            max_delta_d=500,
            seed=42,
            estimation_function=lambda x, y: estimation.estimator_exp_gaussian(x, y, ct=.1, cd=50, frac_bg=None),
            trigger_kde_kwargs=trigger_kde_kwargs,
            bg_kde_kwargs=bg_kde_kwargs,
            remove_coincident_pairs=False
        )
        vb_sepp_iso = validate.SeppValidationFixedModel(data,
                                                    sepp_iso,
                                                    data_index=cid,
                                                    spatial_domain=poly,
                                                    cutoff_t=cutoff)
        vb_sepp_iso.set_sample_units(vb_s.roc)
        res_sepp_iso = vb_sepp_iso.run(time_step=1, n_iter=100, train_kwargs={'niter': 100})
    except Exception:
        res_sepp_iso = None
    name = 'sepp_iso'
    with open(os.path.join(outdir, name), 'w') as f:
        dill.dump(res_sepp_iso, f)

    all_res = {
        'sepp': res_sepp,
        'sepp_refl': res_sepp_refl,
        'sepp_iso': res_sepp_iso,
        'kde_s': res_s,
        'kde_st': res_st
    }

    # with open('%s_validation.pickle' % name, 'w') as f:
    #     dill.dump(all_res, f)

    return all_res


def run_camden_validation():
    outdir = os.path.join(OUT_DIR, 'validation', 'camden')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ## Camden
    cutoff = 212

    poly = cad.get_camden_region(as_shapely=True)
    data, t0, cid = cad.get_crimes_by_type()

    kde_s = hotspot.SKernelHistoricVariableBandwidthNn(60, nn=20)
    vb_s = validation.ValidationIntegration(data,
                                            kde_s,
                                            data_index=cid,
                                            spatial_domain=poly,
                                            cutoff_t=cutoff)
    vb_s.set_sample_units(GRID_LENGTH, n_sample_per_grid=N_PER_GRID)
    camden_res_s = vb_s.run(time_step=1, n_iter=100)
    name = 'skde'
    with open(os.path.join(outdir, name), 'w') as f:
        dill.dump(camden_res_s, f)


    kde_st = hotspot.STLinearSpaceExponentialTime(400., 10.)  # 400m, 10ln2 days half life
    vb_st = validation.ValidationIntegration(data,
                                            kde_st,
                                            data_index=cid,
                                            spatial_domain=poly,
                                            cutoff_t=cutoff)
    vb_st.set_sample_units(GRID_LENGTH, n_sample_per_grid=N_PER_GRID)
    camden_res_st = vb_st.run(time_step=1, n_iter=100)
    name = 'stkde'
    with open(os.path.join(outdir, name), 'w') as f:
        dill.dump(camden_res_st, f)

    try:
        sepp = cad.construct_sepp(data,
                                  max_delta_t=150,
                                  max_delta_d=500,
                                  remove_coincident_pairs=True
                                  )
        vb_sepp = validate.SeppValidationFixedModelIntegration(data,
                                                    sepp,
                                                    data_index=cid,
                                                    spatial_domain=poly,
                                                    cutoff_t=cutoff)
        vb_sepp.set_sample_units(GRID_LENGTH, n_sample_per_grid=N_PER_GRID)
        camden_res_sepp = vb_sepp.run(time_step=1, n_iter=100, train_kwargs={'niter': 100})
    except Exception:
        camden_res_sepp = None
    name = 'sepp'
    with open(os.path.join(outdir, name), 'w') as f:
        dill.dump(camden_res_sepp, f)


    try:
        sepp_mb50 = cad.construct_sepp(data,
                                     max_delta_t=150,
                                     max_delta_d=500,
                                     remove_coincident_pairs=False,
                                     min_bandwidth=[0., 50., 50.]
                                     )
        vb_sepp_mb50 = validate.SeppValidationFixedModelIntegration(data,
                                                    sepp_mb50,
                                                    data_index=cid,
                                                    spatial_domain=poly,
                                                    cutoff_t=cutoff)
        # vb_sepp.set_sample_units(100, n_sample_per_grid=10)
        vb_sepp_mb50.set_sample_units(GRID_LENGTH, n_sample_per_grid=N_PER_GRID)
        camden_res_sepp_mb50 = vb_sepp_mb50.run(time_step=1, n_iter=100, train_kwargs={'niter': 100})
    except Exception:
        camden_res_sepp_mb50 = None
    name = 'sepp_mb50'
    with open(os.path.join(outdir, name), 'w') as f:
        dill.dump(camden_res_sepp_mb50, f)

    try:
        sepp_mb5 = cad.construct_sepp(data,
                                     max_delta_t=150,
                                     max_delta_d=500,
                                     remove_coincident_pairs=False,
                                     min_bandwidth=[0., 5., 5.]
                                     )
        vb_sepp_mb5 = validate.SeppValidationFixedModelIntegration(data,
                                                    sepp_mb5,
                                                    data_index=cid,
                                                    spatial_domain=poly,
                                                    cutoff_t=cutoff)
        # vb_sepp.set_sample_units(100, n_sample_per_grid=10)
        vb_sepp_mb5.set_sample_units(GRID_LENGTH, n_sample_per_grid=N_PER_GRID)
        camden_res_sepp_mb5 = vb_sepp_mb5.run(time_step=1, n_iter=100, train_kwargs={'niter': 100})
    except Exception:
        camden_res_sepp_mb5 = None
    name = 'sepp_mb5'
    with open(os.path.join(outdir, name), 'w') as f:
        dill.dump(camden_res_sepp_mb5, f)

    try:
        sepp_iso = cad.construct_sepp(data,
                                      max_delta_t=150,
                                      max_delta_d=500,
                                      remove_coincident_pairs=False,
                                      # remove_coincident_pairs=True,
                                      sepp_class=pp_models.SeppStochasticNnIsotropicTrigger)
        vb_sepp_iso = validate.SeppValidationFixedModelIntegration(data,
                                                    sepp_iso,
                                                    data_index=cid,
                                                    spatial_domain=poly,
                                                    cutoff_t=cutoff)
        vb_sepp_iso.set_sample_units(GRID_LENGTH, n_sample_per_grid=N_PER_GRID)
        camden_res_sepp_iso = vb_sepp_iso.run(time_step=1, n_iter=100, train_kwargs={'niter': 100})
    except Exception:
        camden_res_sepp_iso = None
    name = 'sepp_iso'
    with open(os.path.join(outdir, name), 'w') as f:
        dill.dump(camden_res_sepp_iso, f)

    all_res = {
        'sepp': camden_res_sepp,
        'kde_s': camden_res_s,
        'kde_st': camden_res_st,
        'sepp_mb50': camden_res_sepp_mb50,
        'sepp_mb5': camden_res_sepp_mb5,
        'sepp_iso': camden_res_sepp_iso
    }

    # with open('camden_validation.pickle', 'w') as f:
    #     dill.dump(all_res, f)

    return all_res