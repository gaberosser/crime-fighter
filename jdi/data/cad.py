from settings import BLUE_DATA
import os
import csv
from jdi.data import consts


def get_borough_names():
    bcs = consts.BOROUGH_CODES
    max_n = 100000
    borough_names = {}
    with open(os.path.join(BLUE_DATA, 'cad.csv')) as f:
        c = csv.DictReader(f)
        i = 0
        while i < max_n:
            if len(bcs) == 0:
                break
            row = c.next()
            code = row['Borough Code']
            name = row['Borough Name']
            if code in bcs:
                bcs.remove(code)
                borough_names[code] = name
            i += 1
    return borough_names
        