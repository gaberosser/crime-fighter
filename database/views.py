from django.shortcuts import render
from django.db.models import Q, Count, Sum, Min, Max
from django.http import HttpResponse
from database import models
import collections
import os
import csv
import numpy as np
import datetime
import pytz
from matplotlib import pyplot as plt

cris_nicl_mapping = {
    'Burglary': [3, 4],
    'Criminal Damage': [10],
    'Drugs': [11],
    'Robbery': [5],
    'Theft & Handling': [6, 7, 8, 13],
    'Violence Against The Person': [1],
}

def month_iterator(start_date, end_date):
    try:
        start_date.tzinfo
    except AttributeError:
        kwargs = {}
    else:
        kwargs = {'tzinfo': pytz.utc}

    this_type = type(start_date)
    sd = start_date
    while sd < end_date:
        next_year = sd.year
        next_month = sd.month + 1
        if sd.month == 12:
            next_year += 1
            next_month = 1
        ed = this_type(next_year, next_month, 1, **kwargs)
        if ed <= end_date:
            yield (sd, ed)
        else:
            yield (sd, end_date+datetime.timedelta(days=1))
        sd = ed


def cris_cad_comparison():
    # get all 'real' crimes from CAD
    cad_qset = models.Cad.objects.exclude(cris_entry__isnull=True).exclude(cris_entry__startswith='NOT')
    # divide into major crime types
    cad_by_type = {}
    cad_monthly = collections.OrderedDict()
    for (k, v) in cris_nicl_mapping.items():
        qry = Q(cl01__in=v) | Q(cl02__in=v) | Q(cl03__in=v)
        cad_by_type[k] = cad_qset.filter(qry)

    start_date = cad_qset.aggregate(m=Min('inc_datetime'))['m'].replace(hour=0, minute=0, second=0)
    end_date = cad_qset.aggregate(m=Max('inc_datetime'))['m']
    for (sd, ed) in month_iterator(start_date, end_date):
        this_month_dict = {}
        for k, v in cad_by_type.items():
            this_month_dict[k] = v.filter(inc_datetime__gte=sd, inc_datetime__lt=ed).values(
                'inc_number',
                'att_map',
                'units_assigned_number',
                'cris_entry',
            ).annotate(cris_count=Count('cris_entry')).count()
        cad_monthly[sd] = this_month_dict


    # aggregate CRIS to borough level
    csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        'opendata/lookup/OA11_LSOA11_MSOA11_LAD11_EW_LUv2.csv'))
    with open(csv_file, 'r') as f:
        c = csv.DictReader(f)
        res = list(c)
    lsoa_codes = list(np.unique([x['LSOA11CD'] for x in res if x['LAD11NM'] == 'Camden']))
    cris_qset = models.Cris.objects.filter(lsoa_code__in=lsoa_codes)

    cris_by_type = {}
    cris_monthly = collections.OrderedDict()
    for k in cris_nicl_mapping.keys():
        cris_by_type[k] = cris_qset.filter(crime_major=k)

    for (sd, ed) in month_iterator(start_date, end_date):
        this_month_dict = {}
        for k, v in cris_by_type.items():
            this_month_dict[k] = v.filter(date__gte=sd.date(), date__lt=ed.date()).aggregate(m=Sum('count'))['m']
        cris_monthly[sd] = this_month_dict

    # zip together
    combined = collections.OrderedDict()
    for sd, v_cad in cad_monthly.items():
        v_cris = cris_monthly[sd]
        d = collections.OrderedDict()
        for k, n_cad in v_cad.items():
            n_cris = v_cris[k]
            d[k] = {'cad': n_cad, 'cris': n_cris}
        combined[sd] = d

    return combined


def cris_cad_plot(combined=None):
    combined = combined or cris_cad_comparison()
    fig, axarr = plt.subplots(3, 2)
    fig.set_size_inches(12, 9)
    dates = [x.date() for x in combined.keys()]
    for i, t in enumerate(cris_nicl_mapping.keys()):
        ax = axarr[i/2, i - 2*(i/2)]
        cris = [x[t]['cris'] for x in combined.values()]
        cad = [x[t]['cad'] for x in combined.values()]
        ax.plot(dates, cris, 'k-x', label='CRIS')
        ax.plot(dates, cad, 'r-x', label='CAD')
        ax.set_title(t)
        ax.set_ylim([0, max(max(cad), max(cris))*1.05])
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)
        if i > 3:
            plt.setp(ax.xaxis.get_majorticklabels(), visible=True, rotation=70)
        if i== 0:
            legend = ax.legend(loc='lower left')



def cris_cad_comparison_view(request):
    combined = cris_cad_comparison()
    return render(request, 'database/cris_cad_compare.html', {'combined': combined},)
