__author__ = 'gabriel'
from database import models
import numpy as np
import cad

s_max = 500  # metres
t_max = 90  # days

camden = cad.get_camden_region()
contract = camden.buffer(-s_max)

target_data, t0, cid = cad.get_crimes_by_type(nicl_type=None)
source_data, tmp2, cid_contract = cad.get_crimes_by_type(nicl_type=None, spatial_domain=contract)

n_full = cid.size
map_contract = np.array([np.any(cid_contract == cid[i]) for i in range(n_full)])
idx_contract = np.arange(n_full)[map_contract]
idx_buffer = np.arange(n_full)[~map_contract]

