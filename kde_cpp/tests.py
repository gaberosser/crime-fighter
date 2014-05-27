__author__ = 'gabriel'
import numpy as np
from kde_cpp import pymvn
from time import time
from kde.methods.pure_python import FixedBandwidthKde

n = 5000
data = np.random.randn(n, 3)
bandwidths = np.array([1., 2., 3.])
p = FixedBandwidthKde(data, bandwidths=bandwidths, normed=True)
c = pymvn.PyFixedBandwidthKde(data, bandwidths, True)

print "C++ pdf external iteration"
tic=time()
resc_it = [c.pdf(data[i]) for i in range(data.shape[0])]
print (time()-tic)

print "C++ pdf internal iteration"
tic=time()
resc = c.pdfm(data)
print (time()-tic)

print "Python/Numpy implementation"
tic=time()
resp=p.pdf(data[:,0], data[:,1], data[:,2])
print time()-tic