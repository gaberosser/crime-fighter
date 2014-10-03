import multiprocessing as mp
import numpy as np
import math
import time
import os


# def compute_something(t):
#     a = 0.
#     for i in range(1000000):
#         a = math.sqrt(t)
#     return a
#
# if __name__ == '__main__':
#
#     pool_size = multiprocessing.cpu_count()
#     os.system('taskset -cp 0-%d %s' % (pool_size, os.getpid()))
#
#     print "Pool size:", pool_size
#     pool = multiprocessing.Pool(processes=pool_size)
#
#     inputs = range(10)
#
#     tic = time.time()
#     builtin_outputs = map(compute_something, inputs)
#     print 'Built-in:', time.time() - tic
#
#     tic = time.time()
#     pool_outputs = pool.map(compute_something, inputs)
#     print 'Pool    :', time.time() - tic

class C(object):

    def __init__(self, i):
        self.i = i

    def run(self):
        return math.log(reduce(lambda x, y: x * y, range(1, self.i + 1)))


class B(object):

    def __init__(self, n):
        self.n = n
        self.kernels = [C(i) for i in range(1, n + 1)]

    def run(self):
        return sum([k.run() for k in self.kernels])


def my_picklable_fun(x):
    return x.run()


class A(object):

    def __init__(self, n):
        self.set_clusters(n)

    def set_clusters(self, n):
        self.clusters = []
        idx = 0
        per_cluster = int(n / mp.cpu_count())
        for i in range(mp.cpu_count() - 1):
            self.clusters.append(B(per_cluster))
            idx += per_cluster
        self.clusters.append(B(n - idx))

    def run(self):
        tic = time.time()
        a = [x.run() for x in self.clusters]
        print a
        print "Took %f seconds" % (time.time() - tic)
        return sum(a)

    def run_parallel(self):
        tic = time.time()
        pool = mp.Pool()
        p = pool.map(my_picklable_fun, self.clusters)

        print p
        print "Took %f seconds" % (time.time() - tic)


