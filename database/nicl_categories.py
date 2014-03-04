__author__ = 'gabriel'
import csv
import collections
import settings
import os
from sandbox import NiclInterpretation

CAD_DATA_DIR = os.path.join(settings.DATA_DIR, 'cad')
NICL_CATEGORIES_CSV = os.path.join(CAD_DATA_DIR, 'nicl_categories.csv')

def add_nicl_data():
    nicl = NiclInterpretation()
