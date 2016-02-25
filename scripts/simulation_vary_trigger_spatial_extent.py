__author__ = 'gabriel'
from point_process import models, simulate, estimation
import numpy as np
import copy
import os
from scripts import OUT_DIR

OUT_DIR = os.path.join(OUT_DIR, 'simulation', 'mohler_vary_spatial_trigger')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
LOCK_FILE = os.path.join(OUT_DIR, 'still_running.lock')
sigmas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
t_total = 1500


def load_results():
    n = len(sigmas)
    sepp = np.empty((n, n), dtype=object)
    sims = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(i, n):
            fn = os.path.join(OUT_DIR, '%.2f_%.2f.pickle' % (sigmas[i], sigmas[j]))
            sepp[i, j] = models.SeppStochasticNn.from_pickle(fn)
            c = simulate.MohlerSimulation()
            c.t_total = 1500
            c.bg_params[0]['sigma'] = [1., 1.]
            c.bg_params[0]['intensity'] = 5
            c.trigger_params['sigma'] = [sigmas[i], sigmas[j]]
            c.run()
            sims[i, j] = copy.deepcopy(c)

    return sepp, sims


def proportion_trigger(sepp):
    func = np.vectorize(lambda x: None if x is None else 1 - x.p.diagonal().sum()/float(x.ndata))
    return func(sepp)


if __name__ == "__main__":

    open(LOCK_FILE, 'a').close()

    try:

        num_iter = 50
        max_delta_t = 100
        max_delta_d = 0.5
        bg_kde_kwargs = {
            'strict': False,
        }

        trigger_kde_kwargs = {
            'strict': False,
        }
        s0, s1 = np.meshgrid(
            sigmas, sigmas
        )

        try:
            sepp_obj
        except NameError:
            sepp_obj = {}  # already started

        for i in range(s0.size):
            tt = sorted([s0.flat[i], s1.flat[i]])
            if tuple(tt) in sepp_obj:
                continue
            c = simulate.MohlerSimulation()
            c.t_total = t_total
            c.bg_params[0]['sigma'] = [1., 1.]
            c.bg_params[0]['intensity'] = 5
            c.trigger_sigma = list(tt)
            init_est = lambda d, t: estimation.estimator_exp_gaussian(d, t, ct=0.1, cd=np.mean(tt))
            c.run()
            data = c.data
            r = models.SeppStochasticNn(data=data,
                                        max_delta_d=max_delta_d,
                                        max_delta_t=max_delta_t,
                                        estimation_function=init_est,
                                        seed=42,
                                        bg_kde_kwargs=bg_kde_kwargs,
                                        trigger_kde_kwargs=trigger_kde_kwargs
            )
            try:
                _ = r.train(niter=num_iter)
            except Exception:
                continue
            sepp_obj[tuple(tt)] = copy.deepcopy(r)

            fullfile = os.path.join(OUT_DIR, '%.2f_%.2f.pickle' % (tt[0], tt[1]))
            r.pickle(fullfile)
    finally:
        os.remove(LOCK_FILE)