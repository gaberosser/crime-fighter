__author__ = 'gabriel'
from point_process import models, simulate, estimation
import numpy as np
import copy

num_iter = 50
max_delta_t = 100
max_delta_d = 0.5

sigmas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
s0, s1 = np.meshgrid(
    sigmas, sigmas
)

try:
    sepp_obj
except NameError:
    sepp_obj = {}  # already started

for i in range(s0.size):
    tt = [s0.flat[i], s1.flat[i]]
    if tuple(tt) in sepp_obj:
        continue
    c = simulate.MohlerSimulation()
    c.t_total = 1500
    c.bg_params[0]['sigma'] = [1., 1.]
    c.bg_params[0]['intensity'] = 5
    c.trigger_params['sigma'] = list(tt)
    init_est = lambda d, t: estimation.estimator_exp_gaussian(d, t, ct=0.1, cd=np.mean(tt))
    c.run()
    data = c.data[:, :3]
    r = models.SeppStochasticNn(data=data,
                                max_delta_d=max_delta_d,
                                max_delta_t=max_delta_t,
                                estimation_function=init_est,
                                seed=42,
    )
    try:
        _ = r.train(niter=num_iter)
    except Exception:
        continue
    sepp_obj[tuple(tt)] = copy.deepcopy(r)