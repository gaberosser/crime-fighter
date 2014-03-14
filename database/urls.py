__author__ = 'gabriel'
from django.conf.urls import patterns, url
from database import views

urlpatterns = patterns('',
    url(r'^$', views.cris_cad_comparison_view, name='cad_cris_comparison')
)