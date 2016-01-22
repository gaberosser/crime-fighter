from . import models
import numpy as np
import logging


class RollingOrigin(object):

    def __init__(self, data,
                 data_index=None,
                 initial_cutoff_t=None,
                 dt=1.,
                 dt_plus=1.,
                 dt_minus=0.,
                 data_class=None):

        """
        :param data:
        :param data_index:
        :param initial_cutoff_t: Optionally supply the first cutoff time, otherwise the approximate midway point is used
        :param dt: The time interval by which the cutoff is advanced each iteration
        :param dt_plus: The upper time range of the TESTING data, expressed relative to the cutoff time.
        If this is None, all testing data are used (subject to dt_minus)
        beyond the cutoff are used.
        :param dt_minus: The lower time range of the TESTING data, expressed relative to the cutoff time.
        Default is 0, meaning that all data >= cutoff are used (subject to dt_plus).
        :param data_class: Optionally provide a class to contain the data. This should be derived from
        SpaceTimeDataArray, or mimic that class.
        If no data_class is provided, we either use a generic default or the existing data class.
        """
        # set default parameters in the event they are supplied as None
        if dt is None:
            dt = 1.
        if dt_plus is None:
            dt_plus = 1.
        if dt_minus is None:
            dt_minus = 0.

        if data_class is None:
            if isinstance(data, models.DataArray):
                self.data_class = data.__class__
            else:
                self.data_class = models.SpaceTimeDataArray
        else:
            self.data_class = data_class

        self.data = None
        self.data_index = None
        self.set_data(data, data_index=data_index)

        assert dt > 0., "dt must be positive and non-zero"
        assert dt_minus >= 0., "dt_minus must be positive"
        if dt_plus:
            assert dt_plus >= 0., "dt_plus must be positive"
        assert dt_plus > dt_minus, "Require that dt_plus > dt_minus"

        self.dt = dt
        self.dt_plus = dt_plus
        self.dt_minus = dt_minus

        # set initial time cut point
        self.cutoff_t = initial_cutoff_t or self.t[int(self.ndata / 2)]

        # set logger
        self.logger = logging.getLogger(name=self.__class__.__name__)

    def set_data(self, data, data_index=None):
        # sort data in increasing time
        self.data = self.data_class(data)
        sort_idx = np.argsort(self.t)
        self.data = self.data.getrows(sort_idx)
        self.data_index = data_index
        if data_index is not None:
            self.data_index = np.array(data_index)
            self.data_index = self.data_index[sort_idx]

    @property
    def ndata(self):
        return self.data.ndata

    @property
    def t(self):
        return self.data.time.toarray()

    @property
    def training(self):
        return self.data.getrows(self.t < self.cutoff_t)

    @property
    def testing_index(self):
        """
        Return the array indices corresponding to the testing data based on the supplied parameters.
        :return: Indices corresponding to the testing data
        """
        bottom = self.cutoff_t + self.dt_minus
        if self.dt_plus is None:
            ind = self.t >= bottom
        else:
            ind = (self.t >= bottom) & (self.t < (self.cutoff_t + self.dt_plus))
        return np.where(ind)[0]

    @property
    def testing(self):
        """
        :return: Testing data for comparison with predictions, based on value of cutoff_t.
        """
        return self.data.getrows(self.testing_index)

    @property
    def testing_data_index(self):
        """
        :return: The data indices attached to the testing data.  If self.data_index has not been set, return the lookup
        indices instead.
        """
        ind = self.testing_index
        if self.data_index is not None:
            return self.data_index[ind]
        else:
            return ind

    def advance(self, dt=None):
        dt = dt or self.dt
        self.cutoff_t += dt

    def iterator(self, niter=None):
        """
        Iterate over the cutoff times and yield a reference to the object each time
        :param niter: Optionally fix the number of iterations
        :return:
        """
        tmax = self.t.max()
        max_iter = int((tmax - self.cutoff_t - self.dt_minus) // self.dt)
        if niter is not None:
            # check that the number of iterations will be possible
            assert niter <= max_iter, "the requested number of iterations is too high"
        else:
            niter = max_iter
        def generator():
            self.logger.info("Starting iterations, %d in total", niter)
            for i in range(niter):
                self.logger.info("Iteration %d, cutoff_t is %f", i, self.cutoff_t)
                yield self
                self.advance()
        return generator()
