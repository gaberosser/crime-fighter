## Apply analysis functions to data
library('stpp')
library('rpanel')
source('fetchData.R')

cadSTIK <- function(crime_type=3) {
  
}
crime_type=3
camden <- getBufferedCamden(20.0)
camden_poly <- camden@polygons[[1]]@Polygons[[1]]@coords[nrow(camden_poly):1,]
cad <- getCad(crime_type)
xy <- apply(array(cad$pt), MARGIN=1, FUN=coordinates)
x <- xy[1,]
y <- xy[2,]
min_t <- as.integer(format(min(cad$inc_datetime), "%s"))
t <- (as.integer(format(cad$inc_datetime, '%s')) - min_t) / (24*60*60) # days

cad <- as.data.frame(cbind(x, y, t))
## TODO: need to JIGGLE points, rather than remove, this is just proof of principal
tmp <- unique(cad[,c('x', 'y')])
cad <- as.3dpoints(cad[rownames(tmp),])

# bandwidth (m)
h = 200

Mt <- density(cad[,3], n=1000) # estimate of first-order time-component (normalised)
mut <- Mt$y[findInterval(cad[, 3], Mt$x)]

Ms <- kernel2d(cad[, 1:2], camden_poly, h=h, nx=1000, ny=1000)
atx <- findInterval(cad[, 1], Ms$x)
aty <- findInterval(cad[, 2], Ms$y)
mhat <- NULL

for (i in 1:length(atx)) {
  mhat <- c(mhat, Ms$z[atx[i], aty[i]])
}
u <- seq(0, 50, 1) # delta distance, m
v <- seq(0, 200, 7) # delta time, days
stik <- STIKhat(xyt=cad, s.region=camden_poly, t.region=c(1, max(cad[,3])), lambda=mhat*mut, dist=u, times=v, infectious=TRUE)
g <- PCFhat(xyt=cad, s.region=camden_poly, t.region=c(1, max(cad[,3])), lambda=mhat*mut, dist=u, times=v)