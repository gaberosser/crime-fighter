__author__ = 'gabriel'
import psycopg2
from settings import settings

try:
    conn = psycopg2.connect("dbname='%s' user='%s' host='%s' password='%s'" % (settings.POSTGRES_DB, settings.POSTGRES_USER, settings.POSTGRES_HOST, settings.POSTGRES_PASS))
except Exception:
    print "I am unable to connect to the database"