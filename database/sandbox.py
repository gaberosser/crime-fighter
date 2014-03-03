__author__ = 'gabriel'
import psycopg2
import collections
from settings import settings


def connect():
    try:
        conn = psycopg2.connect("dbname='%s' user='%s' host='%s' password='%s'" % (settings.POSTGRES_DB, settings.POSTGRES_USER, settings.POSTGRES_HOST, settings.POSTGRES_PASS))
    except Exception:
        print "I am unable to connect to the database"
        raise psycopg2.DatabaseError()

    return conn


def create_table(conn, name='mytable'):
    cur = conn.cursor()
    qry = \
        """CREATE TABLE %s (
        mystr varchar(128) NOT NULL,
        myint int,
        mydouble double precision
        );
        SELECT AddGeometryColumn('%s', 'mypoint', 3395, 'POINT', 2);
        """
    try:
        cur.execute(qry % (name, name))
    except Exception as e:
        print repr(e)
        conn.rollback()
        raise


def drop_table(conn, name):
    cur = conn.cursor()
    qry = \
        """DROP TABLE %s;"""
    try:
        cur.execute(qry % name)
    except Exception as e:
        print repr(e)
        conn.rollback()
        raise
    else:
        conn.commit()


def show_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    return cur.fetchall()


class PostgresTable(object):
    conn = connect()
    name = 'mytable'
    cols = []

    def __init__(self):
        self.create_table()

    @property
    def create_query(self):
        qry = "CREATE TABLE IF NOT EXISTS %s (\n" % self.name
        for (i, x) in enumerate(self.cols):
            qry += (',\n' if i > 0 else '') + x
        qry += ");\n"
        return qry

    def create_table(self):
        cur = self.conn.cursor()
        try:
            cur.execute(self.create_query)
        except Exception as e:
            print repr(e)
            self.conn.rollback()
            raise
        else:
            self.conn.commit()


class PostgisTable(PostgresTable):
    geom_cols = {}
    srid = 3395
    name = 'mypostgistable'

    @property
    def create_query(self):
        """ Add geometry columns to the create query """
        qry = super(PostgisTable, self).create_query
        for k, v in self.geom_cols.values():
            qry += "SELECT AddGeometryColumn('%s', '%s', %u, '%s', %u);\n" % (self.name, k, self.srid, v['type'], v['dim'])
        return qry


class NiclInterpretation(PostgresTable):
    name = 'nicl_category'
    cols = [
        'number integer PRIMARY KEY',
        'description varchar(128)',
        'cat_no integer',
        'cat_letter varchar(3)'
    ]


class CadData(PostgisTable):
    cols = [
        'OP01 integer REFERENCES nicl_category (number)',
        'OP02 integer REFERENCES nicl_category (number)',
        'OP03 integer REFERENCES nicl_category (number)',
    ]
    geom_cols = collections.OrderedDict([
        ('att_map', {'type': 'POINT', 'dim': 2}),
        ('inc_map', {'type': 'POINT', 'dim': 2}),
        ('call_map', {'type': 'POINT', 'dim': 2}),
    ])
