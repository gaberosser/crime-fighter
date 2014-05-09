import multiprocessing
import numpy as np
import math
import time
import os


def compute_something(t):
    a = 0.
    for i in range(10000000):
        a = math.sqrt(t)
    return a

if __name__ == '__main__':

    pool_size = multiprocessing.cpu_count()
    os.system('taskset -cp 0-%d %s' % (pool_size, os.getpid()))

    print "Pool size:", pool_size
    pool = multiprocessing.Pool(processes=pool_size)
    
    inputs = range(10)

    tic = time.time()
    builtin_outputs = map(compute_something, inputs)
    print 'Built-in:', time.time() - tic

    tic = time.time()
    pool_outputs = pool.map(compute_something, inputs)
    print 'Pool    :', time.time() - tic