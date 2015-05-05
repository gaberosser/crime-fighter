__author__ = 'gabriel'
import numpy as np
import operator
from contextlib import closing
from functools import partial
import ctypes
from itertools import izip
import multiprocessing as mp
from kde import kernels
from stats.logic import weighted_stdev
from sklearn.neighbors import NearestNeighbors
from data.models import Data, DataArray, SpaceTimeDataArray, CartesianSpaceTimeData, negative_time_dimension
import warnings
import logging


logger = logging.getLogger(__name__)
# default: output all logs to console
ch = logging.StreamHandler()
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)


def runner_pdf(x, **kwargs):
    kde, target_data = x
    if kde is None:
        return
    return kde.pdf(target_data, **kwargs)


def runner_marginal_pdf(x, **kwargs):
    kde, target_data = x
    if kde is None:
        return
    return kde.marginal_pdf(target_data, **kwargs)


def runner_partial_marginal_pdf(x, **kwargs):
    kde, target_data = x
    if kde is None:
        return
    return kde.partial_marginal_pdf(target_data, **kwargs)


def runner_marginal_cdf(x, **kwargs):
    kde, target_data = x
    if kde is None:
        return
    return kde.marginal_cdf(target_data, **kwargs)


runners = {
    'pdf': runner_pdf,
    'marginal_pdf': runner_marginal_pdf,
    'partial_marginal_pdf': runner_partial_marginal_pdf,
    'marginal_cdf': runner_marginal_cdf,
}


class MultiKde(object):
    def __init__(self,
                 parallel=True,
                 **kwargs
                 ):
        self.kde_array = []
        self.parallel = parallel
        try:
            self.ncpu = kwargs.pop('ncpu', mp.cpu_count())
        except NotImplementedError:
            self.ncpu = 1

    def __getitem__(self, item):
        return self.kde_array[item]

    def add_kde(self, kde):
        """
        Add the supplied KDE to the array.
        kde may be set to None, in which case it is skipped at evaluation time
        :param kde:
        :return:
        """
        self.kde_array.append(kde)

    @property
    def nkde(self):
        return len(self.kde_array)

    def pdf(self, data_array, same_data=False, **kwargs):
        return self._iterative_operation('pdf', data_array, same_data=same_data, **kwargs)

    def marginal_pdf(self, data_array, same_data=False, **kwargs):
        return self._iterative_operation('marginal_pdf', data_array, same_data=same_data, **kwargs)

    def marginal_cdf(self, data_array, same_data=False, **kwargs):
        return self._iterative_operation('marginal_cdf', data_array, same_data=same_data, **kwargs)

    def partial_marginal_pdf(self, data_array, same_data=False, **kwargs):
        return self._iterative_operation('partial_marginal_pdf', data_array, same_data=same_data, **kwargs)

    def _iterative_operation(self, funcstr, target, same_data=False, **kwargs):
        func = runners[funcstr]
        if same_data:
            target = [target] * self.nkde
        elif len(target) != self.nkde:
            raise AttributeError("Length of array of data inputs does not equal the number of KDEs.")
        if not self.parallel:
            z = [func(t, **kwargs) for t in izip(self.kde_array, target)]
        else:
            with closing(mp.Pool(processes=self.ncpu)) as pool:
                async_res = pool.map_async(partial(func, **kwargs), izip(self.kde_array, target))
                z = async_res.get(1e100)
        return z
