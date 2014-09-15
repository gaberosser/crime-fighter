__author__ = 'gabriel'
from analysis import cad
import numpy as np
import dill

res_nicl3_1day_g100_sepp, vb_nicl3_1day_g100_sepp = cad.validate_point_process(nicl_type=3, grid=100, pred_dt_plus=1)
res_nicl3_2day_g100_sepp, vb_nicl3_2day_g100_sepp = cad.validate_point_process(nicl_type=3, grid=100, pred_dt_plus=2)
res_nicl3_1day_g20_sepp, vb_nicl3_1day_g20_sepp = cad.validate_point_process(nicl_type=3, grid=20, pred_dt_plus=1)
res_nicl3_2day_g20_sepp, vb_nicl3_2day_g20_sepp = cad.validate_point_process(nicl_type=3, grid=20, pred_dt_plus=2)
res_nicl3_1day_g100_nnkde, vb_nicl3_1day_g100_nnkde = cad.validate_historic_kde(nicl_type=3, grid=100, pred_dt_plus=1)
res_nicl3_2day_g100_nnkde, vb_nicl3_2day_g100_nnkde = cad.validate_historic_kde(nicl_type=3, grid=100, pred_dt_plus=2)  #
res_nicl3_1day_g20_nnkde, vb_nicl3_1day_g20_nnkde = cad.validate_historic_kde(nicl_type=3, grid=20, pred_dt_plus=1)  #
res_nicl3_2day_g20_nnkde, vb_nicl3_2day_g20_nnkde = cad.validate_historic_kde(nicl_type=3, grid=20, pred_dt_plus=2)  #

res_nicl1_1day_g100_sepp, vb_nicl1_1day_g100_sepp = cad.validate_point_process(nicl_type=1, grid=100, pred_dt_plus=1)
res_nicl1_2day_g100_sepp, vb_nicl1_2day_g100_sepp = cad.validate_point_process(nicl_type=1, grid=100, pred_dt_plus=2)
res_nicl1_1day_g20_sepp, vb_nicl1_1day_g20_sepp = cad.validate_point_process(nicl_type=1, grid=20, pred_dt_plus=1)
res_nicl1_2day_g20_sepp, vb_nicl1_2day_g20_sepp = cad.validate_point_process(nicl_type=1, grid=20, pred_dt_plus=2)
res_nicl1_1day_g100_nnkde, vb_nicl1_1day_g100_nnkde = cad.validate_historic_kde(nicl_type=1, grid=100, pred_dt_plus=1)
res_nicl1_2day_g100_nnkde, vb_nicl1_2day_g100_nnkde = cad.validate_historic_kde(nicl_type=1, grid=100, pred_dt_plus=2)  #
res_nicl1_1day_g20_nnkde, vb_nicl1_1day_g20_nnkde = cad.validate_historic_kde(nicl_type=1, grid=20, pred_dt_plus=1)  #
res_nicl1_2day_g20_nnkde, vb_nicl1_2day_g20_nnkde = cad.validate_historic_kde(nicl_type=1, grid=20, pred_dt_plus=2)  #

res_nicl67_1day_g100_sepp, vb_nicl67_1day_g100_sepp = cad.validate_point_process(nicl_type=[6, 7], grid=100, pred_dt_plus=1)
res_nicl67_2day_g100_sepp, vb_nicl67_2day_g100_sepp  = cad.validate_point_process(nicl_type=[6, 7], grid=100, pred_dt_plus=2)
res_nicl67_1day_g20_sepp, vb_nicl67_1day_g20_sepp = cad.validate_point_process(nicl_type=[6, 7], grid=20, pred_dt_plus=1)
res_nicl67_2day_g20_sepp, vb_nicl67_2day_g20_sepp = cad.validate_point_process(nicl_type=[6, 7], grid=20, pred_dt_plus=2)
res_nicl67_1day_g100_nnkde, vb_nicl67_1day_g100_nnkde = cad.validate_historic_kde(nicl_type=[6, 7], grid=100, pred_dt_plus=1)
res_nicl67_2day_g100_nnkde, vb_nicl67_2day_g100_nnkde = cad.validate_historic_kde(nicl_type=[6, 7], grid=100, pred_dt_plus=2)  #
res_nicl67_1day_g20_nnkde, vb_nicl67_1day_g20_nnkde = cad.validate_historic_kde(nicl_type=[6, 7], grid=20, pred_dt_plus=1)  #
res_nicl67_2day_g20_nnkde, vb_nicl67_2day_g20_nnkde = cad.validate_historic_kde(nicl_type=[6, 7], grid=20, pred_dt_plus=2)  #

f = open('cad_validation.pickle', 'w')

dill.dump((res_nicl3_1day_g100_sepp, vb_nicl3_1day_g100_sepp), f)
dill.dump((res_nicl3_2day_g100_sepp, vb_nicl3_2day_g100_sepp), f)
dill.dump((res_nicl3_1day_g20_sepp, vb_nicl3_1day_g20_sepp), f)
dill.dump((res_nicl3_2day_g20_sepp, vb_nicl3_2day_g20_sepp), f)
dill.dump((res_nicl3_1day_g100_nnkde, vb_nicl3_1day_g100_nnkde), f)
dill.dump((res_nicl3_2day_g100_nnkde, vb_nicl3_2day_g100_nnkde), f)
dill.dump((res_nicl3_1day_g20_nnkde, vb_nicl3_1day_g20_nnkde), f)
dill.dump((res_nicl3_2day_g20_nnkde, vb_nicl3_2day_g20_nnkde), f)
dill.dump((res_nicl1_1day_g100_sepp, vb_nicl1_1day_g100_sepp), f)
dill.dump((res_nicl1_2day_g100_sepp, vb_nicl1_2day_g100_sepp), f)
dill.dump((res_nicl1_1day_g20_sepp, vb_nicl1_1day_g20_sepp), f)
dill.dump((res_nicl1_2day_g20_sepp, vb_nicl1_2day_g20_sepp), f)
dill.dump((res_nicl1_1day_g100_nnkde, vb_nicl1_1day_g100_nnkde), f)
dill.dump((res_nicl1_2day_g100_nnkde, vb_nicl1_2day_g100_nnkde), f)
dill.dump((res_nicl1_1day_g20_nnkde, vb_nicl1_1day_g20_nnkde), f)
dill.dump((res_nicl1_2day_g20_nnkde, vb_nicl1_2day_g20_nnkde), f)
dill.dump((res_nicl67_1day_g100_sepp, vb_nicl67_1day_g100_sepp), f)
dill.dump((res_nicl67_2day_g100_sepp, vb_nicl67_2day_g100_sepp ), f)
dill.dump((res_nicl67_1day_g20_sepp, vb_nicl67_1day_g20_sepp), f)
dill.dump((res_nicl67_2day_g20_sepp, vb_nicl67_2day_g20_sepp), f)
dill.dump((res_nicl67_1day_g100_nnkde, vb_nicl67_1day_g100_nnkde), f)
dill.dump((res_nicl67_2day_g100_nnkde, vb_nicl67_2day_g100_nnkde), f)
dill.dump((res_nicl67_1day_g20_nnkde, vb_nicl67_1day_g20_nnkde), f)
dill.dump((res_nicl67_2day_g20_nnkde, vb_nicl67_2day_g20_nnkde), f)

f.close()