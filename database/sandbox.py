__author__ = 'gabriel'

import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import Polygon, MultiPolygon
from database import models
import cartopy.io.shapereader as shpreader

def plot_boroughs():

    boroughs = models.Division.objects.filter(type='borough')
    ax = plt.axes([0, 0, 1, 0.95],
                  projection=ccrs.OSGB())

    ax.set_extent([497000, 572000, 146400, 208000], ccrs.OSGB())

    # to get the effect of having just the states without a map "background"
    # turn off the outline and background patches
    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)

    plt.title('London boroughs')

    for b in boroughs:
        # pick a default color for the land with a black outline,
        # this will change if the storm intersects with our track
        facecolor = [0.9375, 0.9375, 0.859375]
        edgecolor = 'black'
        polys = [Polygon(x[0]) for x in b.mpoly.coords]
        geom = MultiPolygon(polys)
        ax.add_geometries([geom], ccrs.OSGB(),
                  facecolor=facecolor, edgecolor=edgecolor)

    plt.show()
