__author__ = 'gabriel'
from point_process import validate, models, simulate, plotting, estimation, plotting
import numpy as np
import datetime
from analysis import plotting, chicago
from scipy import sparse


start_date=datetime.datetime(2004, 3, 1, 0)
end_date=datetime.datetime(2005, 3, 1, 0)
res, t0 = chicago.get_crimes_by_type(crime_type='burglary', datetime__gte=start_date, datetime__lt=end_date)
