__author__ = 'gabriel'
from analysis import chicago, plotting
from database import models
import numpy as np
from matplotlib import pyplot as plt

southwest = models.ChicagoDivision.objects.get(name='Southwest').mpoly
south = models.ChicagoDivision.objects.get(name='South').mpoly
west = models.ChicagoDivision.objects.get(name='West').mpoly

southwest_buf = southwest.buffer(1500)
south_buf = south.buffer(1500)
west_buf = west.buffer(1500)

r_chic, ps_chic = chicago.apply_point_process()
r_sw, ps_sw = chicago.apply_point_process(domain=southwest_buf)
r_s, ps_s = chicago.apply_point_process(domain=south_buf)
r_w, ps_w = chicago.apply_point_process(domain=west_buf)

# heatmap of BG: whole region and subregions
