from django.contrib.gis import admin
import models

# Register your models here.
# admin.site.register(models.Borough, admin.GeoModelAdmin)
admin.site.register(models.Borough, admin.OSMGeoAdmin)