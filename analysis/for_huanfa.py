__author__ = 'gabriel'
import hotspot
import cad
import roc
import datetime
import ogr
import os
import shapefile

grid_spacing = 25  # distance between centroids in metres

res, t0 = cad.get_crimes_by_type(nicl_type=range(1, 17))
sk = hotspot.SKernelHistoricVariableBandwidthNn(dt=60, nn=15)

# choose final date 2011/11/1 for training -> 60 days prior ends AFTER crossover to new non-snapped data
days_offset = 245
dt_pred = t0.date() + datetime.timedelta(days=days_offset)
training = res[res[:, 0] <= days_offset]
sk.train(training)

# generate grid
camden = cad.get_camden_region()
r = roc.RocSpatialGrid(poly=camden)
r.set_grid(grid_spacing)

# prediction on grid
z = sk.predict(r.sample_points)

out_shp = 'risk_surface_%s.shp' % dt_pred.strftime('%Y-%m-%d')
w = shapefile.Writer(shapefile.POINT)
w.field('risk', fieldType='N')
for x, y in zip(r.sample_points, z.flat):
    w.point(x[0], x[1])
    w.record(risk=y)
w.save(out_shp)
