library(ks)
library(FNN)

ndim <- 2
ndata <- 1000
nn <- 100

data <- matrix(runif(ndata*ndim), ncol=ndim)
v <- var(data)[c(1, ndim**2)]
std_data <- t(t(data)/v)
nearn <- get.knn(std_data, k=nn)
nndist = nearn$nn.dist[,nn]

bw <- matrix(rep.int(v, times=ndata), nrow=ndata, byrow=TRUE) * nndist

# what now?! no stock way of incorporating variable density KDEs
