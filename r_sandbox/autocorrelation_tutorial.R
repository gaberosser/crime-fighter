library("maptools")
library("spdep")

uk_temp <- as.matrix(read.csv("~/data/uk_avg_temp.csv"))

uk_districts <- readShapePoly(fn="~/data/met_office_regions/met_office_regions",
proj4string=CRS("+prj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 +x_0=400000 +y_0=-100000 +ellps=airy +datum=OSGB36 +units=m +no_defs"))

uk_districts$temp <- t(uk_temp)

EngNE_acf <- acf(x=uk_temp[,1], lag.max=20)
EngNE_pacf <- pacf(x=uk_temp[,1], lag.max=20)

N <- length(uk_districts)

W <- matrix(0, N, N)
rownames(W) <- uk_districts$NAME
colnames(W) <- uk_districts$NAME
W["EngNE", "ScoE"] = 1
W["EngNE", "EA"] = 1
W["EngNE", "EngNW"] = 1
W["EngNE", "Mid"] = 1
W["EA", "EngSE"] = 1
W["EA", "Mid"] = 1
W["EA", "EngNE"] = 1
W["EngSE", "EngSW"] = 1
W["EngSE", "Mid"] = 1
W["EngSE", "EA"] = 1
W["Mid", "EngSW"] = 1
W["Mid", "EngNW"] = 1
W["Mid", "EngNE"] = 1
W["Mid", "EngSE"] = 1
W["Mid", "EA"] = 1
W["EngSW", "EngNW"] = 1
W["EngSW", "Mid"] = 1
W["EngSW", "EngSE"] = 1
W["EngNW", "ScoW"] = 1
W["EngNW", "ScoE"] = 1
W["EngNW", "EngNE"] = 1
W["EngNW", "Mid"] = 1
W["EngNW", "EngSW"] = 1
W["ScoE", "ScoN"] = 1
W["ScoE", "ScoW"] = 1
W["ScoE", "EngNE"] = 1
W["ScoE", "EngNW"] = 1
W["ScoW", "ScoN"] = 1
W["ScoW", "ScoE"] = 1
W["ScoW", "EngNW"] = 1
W["ScoN", "ScoE"] = 1
W["ScoN", "ScoW"] = 1

W <- W/rowSums(W)
Wlist <- mat2listw(W)

lm <- localmoran(x=colMeans(uk_temp), listw=Wlist)
uk_districts$LMI <- lm[,1]
brks <- quantile(as.matrix(lm[,1], , 1), seq(0,1,1/20))
rgb.palette <- colorRampPalette(c("blue", "green","yellow"), space="rgb")
cols <- rgb.palette(20)

plot(uk_districts, col=cols[findInterval(x=uk_districts$LMI, vec=brks, all.inside=TRUE)])
legend(x=7.5e+05, y=1200000, legend=leglabs(brks), fill=cols, cex=0.8, title="Local Moran's I")

