__author__ = 'gabriel'
import psycopg2
import collections
import settings
import os
import csv
import re


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
        self.cursor = self.conn.cursor()
        self.create_table()

    @property
    def num_cols(self):
        return len(self.cols)

    @property
    def table_exists(self):
        all_tables = [x[0] for x in show_tables(self.conn)]
        return self.name in all_tables

    @property
    def create_query(self):
        qry = "CREATE TABLE %s (\n" % self.name
        for (i, x) in enumerate(self.cols):
            qry += (',\n' if i > 0 else '') + x
        qry += ");\n"
        return qry

    def create_table(self, b_force=False):
        if self.table_exists and not b_force:
            return False
        try:
            self.cursor.execute(self.create_query)
        except Exception as e:
            print repr(e)
            self.conn.rollback()
            raise
        else:
            self.conn.commit()
            return True

    def add_row(self, data):
        val_str = '('+','.join(['%s'] * self.num_cols)+')'
        try:
            qry = self.cursor.mogrify('INSERT INTO {0} VALUES {1};'.format(self.name, val_str), data)
            self.cursor.execute(qry)
        except Exception as e:
            import pdb; pdb.set_trace()
            self.conn.rollback()
            raise
        else:
            self.conn.commit()

    def populate(self, it):
        # delete all existing entries
        self.cursor.execute("DELETE FROM {0};".format(self.name))
        self.conn.commit()

        for row in it:
            self.add_row(row)


class PostgisTable(PostgresTable):
    geom_cols = {}
    srid = 3395
    name = 'mypostgistable'

    @property
    def num_cols(self):
        return len(self.cols) + len(self.geom_cols)

    @property
    def create_query(self):
        """ Add geometry columns to the create query """
        cur = self.cursor()
        qry = super(PostgisTable, self).create_query
        for k, v in self.geom_cols.items():
            qry += cur.mogrify("SELECT AddGeometryColumn(%s, %s, %s, %s, %s);", (self.name, k, self.srid, v['type'], v['dim'],))
        return qry


class Ocu(PostgresTable):
    name = 'ocu'
    cols = [
        'code varchar(3) PRIMARY KEY',
        'description varchar(128)',
    ]


class Nicl(PostgresTable):
    name = 'nicl_category'
    cols = [
        'number integer PRIMARY KEY',
        'category varchar(128)',
        'description varchar(128)',
        'cat_no integer',
        'cat_letter varchar(8)'
    ]

    def add_row(self, data):
        # convert empty string to None
        if not data[3]:
            data[3] = None
        super(Nicl, self).add_row(data)


class CadData(PostgisTable):
    name = 'cad'
    cols = [
        'call_sign varchar(8)',
        'res_type varchar(4)',
        'res_own varchar(3) REFERENCES ocu (code)',
        'inc_no integer NOT NULL',
        'inc_date date NOT NULL',
        'inc_day varchar(3) NOT NULL',
        'inc_time time(0)',
        'OP01 integer REFERENCES nicl_category (number)',
        'OP02 integer REFERENCES nicl_category (number)',
        'OP03 integer REFERENCES nicl_category (number)',
        'att_bocu varchar(3) REFERENCES ocu (code)',
        'caller varchar(1)',
        'CI01 integer REFERENCES nicl_category (number)',
        'CI02 integer REFERENCES nicl_category (number)',
        'CI03 integer REFERENCES nicl_category (number)',
        'cris_entry varchar(16)',  # TODO: reference table
        'number_units integer',
        'grade varchar(1)',
        'uc boolean',  # TODO: what is this?
        'arrival_date date',
        'arrival_time time(0)',
        'response_secs NUMERIC(6, 2)'
    ]
    geom_cols = collections.OrderedDict([
        ('att_map', {'type': 'POINT', 'dim': 2}),
        ('inc_map', {'type': 'POINT', 'dim': 2}),
        ('call_map', {'type': 'POINT', 'dim': 2}),
    ])


def setup_ocu():
    CAD_DATA_DIR = os.path.join(settings.DATA_DIR, 'cad')
    OCU_CSV = os.path.join(CAD_DATA_DIR, 'ocu.csv')
    ocu = Ocu()
    with open(OCU_CSV, 'r') as f:
        c = csv.reader(f)
        ocu.populate(c)


def setup_nicl():
    CAD_DATA_DIR = os.path.join(settings.DATA_DIR, 'cad')
    NICL_CATEGORIES_CSV = os.path.join(CAD_DATA_DIR, 'nicl_categories.csv')
    nicl = Nicl()
    with open(NICL_CATEGORIES_CSV, 'r') as f:
        c = csv.reader(f)
        nicl.populate(c)


def parse_cad_rows(csv_reader, srid):
    def generate_point(x, y, srid):
        try:
            x = int(x)
            y = int(y)
        except ValueError:
            return None
        return "ST_GeomFromText('POINT(%u, %u)', %u)" % (x, y, srid)

    for r in csv_reader:
        row = [x.strip() for x in r]
        res = collections.OrderedDict(
            [
                ('call_sign', row[0]),
                ('res_type', row[1]),
                ('res_own', row[2]),
                ('inc_no', int(row[3]) if row[3] else None),
                ('inc_date', row[4]),
                ('inc_day', row[5]),
                ('inc_time', row[6]),
                ('OP01', int(row[7]) if row[7] else None),
                ('OP02', int(row[8]) if row[8] else None),
                ('OP03', int(row[9]) if row[9] else None),
                ('att_bocu', row[10]),
                ('caller', row[17]),
                ('CI01', int(row[18]) if row[18] else None),
                ('CI02', int(row[19]) if row[19] else None),
                ('CI03', int(row[20]) if row[20] else None),
                ('cris_entry', row[21]),
                ('number_units', int(row[22]) if row[22] else None),
                ('grade', row[23]),
                ('uc', row[24] == 'Y'),
                ('arrival_date', row[25]),
                ('arrival_time', row[26]),
                ('response_secs', row[28]),
                ('att_map', generate_point(row[11], row[12], srid)),
                ('inc_map', generate_point(row[13], row[14], srid)),
                ('call_map', generate_point(row[15], row[16], srid)),
                ]
        )
        yield res.values()


def setup_cad():

    CAD_DATA_DIR = os.path.join(settings.DATA_DIR, 'cad')
    CAD_CSV = os.path.join(CAD_DATA_DIR, 'mar2011-mar2012.csv')
    cad = CadData()
    with open(CAD_CSV, 'r') as f:
        c = csv.reader(f, skipinitialspace=True)
        fields = [x.strip() for x in c.next()]
        it = parse_cad_rows(c, cad.srid)
        cad.populate(it)


def setup_all():
    nicl = Nicl()
    nicl.populate()
    ocu = Ocu()
    ocu.populate()
    cad = CadData()
