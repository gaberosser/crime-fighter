__author__ = 'gabriel'
from django.contrib.gis.gdal import DataSource, field
import shapefile

dpath = '/home/gabriel/Downloads/camden_itn/camden_itn_buff100.shp'
ds = DataSource(dpath)
lyr = ds[0]

fields = lyr.fields
values = [lyr.get_fields(x) for x in fields]
geoms = lyr.get_geoms()

w = shapefile.Writer(shapefile.POLYLINE)

for i, f in enumerate(fields):
    if lyr.field_types[i] is field.OFTString:
        w.field(f, 'C', 128)
    elif lyr.field_types[i] is field.OFTReal:
        w.field(f, 'N')


for i in range(len(geoms)):
    this_coords = [x[:2] for x in geoms[i].coords]
    w.line([this_coords])
    w.record(*[x[i] for x in values])

w.save('sarahs_shapefile')
