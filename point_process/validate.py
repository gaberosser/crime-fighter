__author__ = 'gabriel'
import numpy as np
import math
import runner
from analysis import validation
import models


class PpValidation(validation.ValidationBase):

    pp_class = models.PointProcess # model class

    def __init__(self, data, spatial_domain=None, grid_length=None, tmax_initial=None, model_args=None, model_kwargs=None):
        """ Thin wrapper for parent's init method, but pp model class is set """
        super(PpValidation, self).__init__(data, self.pp_class, spatial_domain=spatial_domain, grid_length=grid_length,
                                           tmax_initial=tmax_initial, model_args=model_args, model_kwargs=model_kwargs)

    def set_t_cutoff(self, cutoff_t, **kwargs):
        """ Disable training the model by default to allow greater control of the initial estimate for p """
        self.cutoff_t = cutoff_t

    def run(self, dt, t_upper=None, niter=20, **kwargs):
        """
        Run the required train / predict / assess sequence
        """
        t0 = self.cutoff_t
        t = dt
        t_upper = min(t_upper or np.inf, self.testing()[-1, 0])

        cfrac = []
        carea = []
        pai = []
        polys = []
        ps = []

        if t0 >= t_upper:
            return polys, ps, carea, cfrac, pai

        # initial training
        self.train_model(niter=niter, **kwargs)

        try:
            while self.cutoff_t < t_upper:
                # store P matrix for later analysis
                ps.append(self.model.p)
                # predict and assess
                this_polys, _, this_carea, this_cfrac, this_pai = self.predict_assess(dt_plus=dt, dt_minus=0, **kwargs)
                carea.append(this_carea)
                cfrac.append(this_cfrac)
                pai.append(this_pai)
                polys.append(this_polys)
                pre_training = self.training
                self.set_t_cutoff(self.cutoff_t + dt)
                # update p based on previous
                self.model.p = self.compute_new_p(pre_training)

                self.train_model(niter=niter, **kwargs)

        finally:
            self.set_t_cutoff(t0)

        return polys, ps, carea, cfrac, pai

    def compute_new_p(self, pre_training):
        """ Compute the new initial estimate of p based on the previous value.
        Assumes that the new training set is the old set with additional records. """
        num_old = len(pre_training)
        num_new = len(self.training)
        if (num_new - num_old) < 0:
            raise AttributeError("More records in previous training set than new training set")
        new_recs = self.training[num_old:]
        pre_p = self.model.p
        if pre_p.shape[0] != len(pre_training):
            raise AttributeError("Model p matrix has incorrect shape")
        # recompute new P using initial estimator
        new_p = self.model.estimator(self.training)
        # restore former probs
        new_p[:num_old, :num_old] = pre_p
        return new_p

if __name__ == "__main__":
    from database import logic, models
    from scipy.stats import multivariate_normal
    from point_process import models as pp_models
    from point_process import simulate
    from analysis import hotspot
    from analysis import plotting
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    camden = models.Division.objects.get(name='Camden')
    xm = 526500
    ym = 186000
    nd = 1000
    # nice normal data
    # data = np.hstack((np.random.normal(loc=5, scale=5, size=(nd, 1)),
    #                   multivariate_normal.rvs(mean=[xm, ym], cov=np.diag([1e5, 1e5]), size=(nd, 1))))

    # moving cluster
    # data = np.hstack((
    #     np.linspace(0, 10, nd).reshape((nd, 1)),
    #     xm + np.linspace(0, 5000, nd).reshape((nd, 1)) + np.random.normal(loc=0, scale=100, size=(nd, 1)),
    #     ym + np.linspace(0, -4000, nd).reshape((nd, 1)) + np.random.normal(loc=0, scale=100, size=(nd, 1)),
    # ))

    # Point process simulated data
    c = simulate.MohlerSimulation()
    c.run()
    data = np.array(c.data)[:, :3]  # (t, x, y, b_is_BG)
    data = data[data[:, 0].argsort()]

    # use Bowers kernel
    # stk = hotspot.STKernelBowers(1, 1e-4)
    # vb = ValidationBase(data, hotspot.Hotspot, camden.mpoly, model_args=(stk,))
    # vb.set_grid(grid_length=200)
    # vb.set_t_cutoff(4.0)

    # use basic historic data spatial hotspot
    # sk = hotspot.SKernelHistoric(2) # use heatmap from final 2 days data
    # vb = ValidationBase(data, hotspot.Hotspot, camden.mpoly, model_args=(sk,))
    # vb.set_grid(grid_length=200)
    # vb.set_t_cutoff(4.0)

    # use Point process learning method
    vb = PpValidation(data, model_kwargs={'max_trigger_t': 80, 'max_trigger_d': 0.75})
    vb.set_grid(grid_length=3)
    polys, ps, carea, cfrac, pai = vb.run(dt=5, t_upper=450, niter=15)

    mc = np.mean(np.array(carea), axis=0)
    mf = np.mean(np.array(cfrac), axis=0)
    mp = np.mean(np.array(pai), axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.array(carea).transpose(), np.array(cfrac).transpose(), color='k', alpha=0.2)
    ax.plot(mc, mf, 'r-')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.array(carea).transpose(), np.array(pai).transpose(), color='k', alpha=0.2)
    ax.plot(mc, mp, 'r-')