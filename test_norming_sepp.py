import datetime
import numpy as np
from point_process import models as pp_models
from point_process import estimation, simulate


def almost_equal(a, b, tol=1e-12):
    return np.all(
        np.abs(a - b) < tol
    )

if __name__ == '__main__':

    pp_class = pp_models.SeppStochasticNnReflected
    # max_delta_d = 0.75
    # max_delta_t = 80
    max_delta_d = 100
    max_delta_t = 100

    bg_kde_kwargs = {
        'number_nn': [101, 16],
    }

    trigger_kde_kwargs = {
        # 'min_bandwidth': [1., .005, .05],
        'number_nn': 15,
    }


    c = simulate.MohlerSimulation()
    # c.off_sigma_x = 1.0
    c.seed(42)
    c.run()
    data = np.array(c.data)  # (t, x, y)
    # sort data by time ascending (may be done already?)
    data = data[data[:, 0].argsort()]

    r0 = pp_class(data=data, max_delta_d=max_delta_d, max_delta_t=max_delta_t,
                                        bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs,
                                        parallel=False)

    p0 = estimation.estimator_bowers(data, r0.linkage, ct=1, cd=10)
    r0.p = p0
    r0.set_seed(42)

    alpha = 0.1

    new_data = data.copy()
    new_data[:, 1:] *= alpha

    r1 = pp_class(data=new_data, max_delta_d=max_delta_d * alpha, max_delta_t=max_delta_t,
                                        bg_kde_kwargs=bg_kde_kwargs, trigger_kde_kwargs=trigger_kde_kwargs,
                                        parallel=False)

    p1 = estimation.estimator_bowers(new_data, r1.linkage, ct=1, cd=10 / alpha)
    r1.p = p1
    r1.set_seed(42)

    print np.all(np.abs(p0.data - p1.data) < 1e-10)

    a0, b0, c0 = r0.sample_data()
    a1, b1, c1 = r1.sample_data()

    print len(a0) == len(a1)

    r0._iterate()
    r1._iterate()

    print r0.num_bg[0] == r1.num_bg[0]
    print almost_equal(r0.trigger_kde.nn_distances, r1.trigger_kde.nn_distances)
    print almost_equal(r0.trigger_kde.bandwidths[:, 1:], r1.trigger_kde.bandwidths[:, 1:] / alpha)

    print np.all(r0.trigger_kde.data.toarray(0) == r1.trigger_kde.data.toarray(0))

    print almost_equal(r0.trigger_kde.data.toarray(1), r1.trigger_kde.data.toarray(1) / alpha)
    print almost_equal(r0.trigger_kde.data.toarray(2), r1.trigger_kde.data.toarray(2) / alpha)

    k0 = r0.trigger_kde.kernel_clusters[0].kernels[0]
    k1 = r1.trigger_kde.kernel_clusters[0].kernels[0]

    print k0.mean[0] == k0.mean[0]
    print almost_equal(k1.mean[1:], k0.mean[1:] * alpha)
    print k0.vars[0] == k0.vars[0]
    print almost_equal(k1.vars[1:], k0.vars[1:] * (alpha ** 2))


    x0 = np.linspace(-1, 1, 100)
    x1 = x0 * alpha
    y0 = r0.trigger_kde.marginal_pdf(x0, dim=1)
    y1 = r1.trigger_kde.marginal_pdf(x1, dim=1)

    print almost_equal(y0, y1 * alpha)

    r0.train(data=data, niter=10)
    r1.train(data=new_data, niter=10)
    print r0.num_bg == r1.num_bg
