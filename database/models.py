from django.contrib.gis.db import models
from django.db.models.signals import pre_save
from django.db import IntegrityError
from django.db import connection

WEEKDAY_CHOICES = (
    ('MON', 'Monday'),
    ('TUE', 'Tuesday'),
    ('WED', 'Wednesday'),
    ('THU', 'Thursday'),
    ('FRI', 'Friday'),
    ('SAT', 'Saturday'),
    ('SUN', 'Sunday'),
)

CALLER_CHOICES = (
    ('V', 'Victim'),
    ('S', 'Staff on duty'),
    ('O', 'Other'),
    ('T', 'Third party'),
    ('W', 'Witness'),
)

GRADE_CHOICES = (
    ('I', 'Immediate'),
    ('S', 'Soonest'),
    ('E', 'Extended'),
    ('R', 'Referred'),
    ('P', 'Police generated'),
)

## standalone (non-Django) table class
## TODO: connection is currently specified by Django settings, but can even eliminate this if required.

def where_query_from_dict(where_dict):
    """
    Generate a postgresql WHERE subquery
    Operators are indicated by a preceding * character
    e.g. *LIKE 'foo%'
    e.g. *IS NOT NULL
    :param where_dict:
    :return: query string
    """
    qry_list = []
    for k, v in where_dict.items():
        if v is None:
            # replace with NULL for SQL
            v = 'NULL'
        if not isinstance(v, str):
            qry_list.append("{0} = {1}".format(k, v))
        elif '*' in v:
            # operator present - strip out all * characters
            rhs = v.replace('*', '')
            qry_list.append("{0} {1}".format(k, rhs))
        else:
            qry_list.append("{0} = {1}".format(k, v))
    return "WHERE {0}".format(' AND '.join(qry_list))


class GeosTable(object):

    schema = None
    schema_name = 'public'
    table_name = None
    dependencies = None
    spatial_indices = None

    def __init__(self):
        self.cur = connection.cursor().cursor
        self.bind_or_create()

    @property
    def pre_init_query(self):
        return None

    @property
    def post_init_query(self):
        # add a function that avoids the divide by zero issue
        return """
        CREATE OR REPLACE FUNCTION divide(num float, den float) RETURNS float AS $$
            BEGIN
                RETURN CASE WHEN den > 0 THEN num / den ELSE NULL END;
            END;
        $$ LANGUAGE plpgsql;
        """

    def create_spatial_index(self):
        if self.spatial_indices is None:
            return
        qry = ''
        for ix, col in self.spatial_indices.items():
            qry = """
            DO $$
            BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM   pg_class c
                JOIN   pg_namespace n ON n.oid = c.relnamespace
                WHERE  c.relname = '{0}'
                AND    n.nspname = '{1}' -- 'public' by default
                ) THEN
                CREATE INDEX {0} ON {1}.{2} USING GIST ({3});
            END IF;

            END$$;
            """.format(
                ix,
                self.schema_name,
                self.table_name,
                col
            )
        self.cur.execute(qry)

    def cluster_by_index(self, idx=0):
        """
        Cluster the data in the table using the index specified by idx
        :param idx: Default=0, the lookup index for the db index to use (only use in the event that >1 index exists)
        """
        if self.spatial_indices is None:
            raise AttributeError("No spatial indices have been declared, unable to cluster data.")
        qry = """
        CLUSTER {0} USING {1};
        """.format(
            self.table_name,
            self.spatial_indices.keys()[idx]
        )
        self.cur.execute(qry)

    def recreate_spatial_index(self):
        if self.spatial_indices is None:
            return
        qry = ''
        for ix in self.spatial_indices.keys():
            qry += """
            DROP INDEX IF EXISTS {0}
            """.format(ix)
        self.cur.execute(qry)
        self.create_spatial_index()

    def bind_or_create(self):
        # test for table
        self.cur.execute(
            "SELECT TRUE FROM pg_tables WHERE tablename = '{0}'".format(self.table_name)
        )

        if self.cur.fetchone() is None:
            # table does not exist
            if self.pre_init_query:
                self.cur.execute(self.pre_init_query)

            if self.dependencies:
                for dep in self.dependencies:
                    # instantiate in order to check table existence
                    obj = dep()

            qry = """
            CREATE TABLE {0}({1});
            """.format(self.table_name, ','.join(self.schema))
            self.cur.execute(qry)

            self.create_spatial_index()

            if self.post_init_query:
                self.cur.execute(self.post_init_query)

    def rewrite_table(self):
        qry = """
        DROP TABLE {0} CASCADE;
        """.format(self.table_name)
        self.cur.execute(qry)
        self.bind_or_create()

    def truncate_table(self):
        qry = """
        TRUNCATE TABLE {0} CASCADE;
        """.format(self.table_name)
        self.cur.execute(qry)

    @property
    def fields(self):
        qry = """
        SELECT * FROM {0} WHERE FALSE;
        """.format(self.table_name)
        self.cur.execute(qry)
        return [x.name for x in self.cur.description]

    def results_to_dict(self, res, fields=None):
        """
        Convert the raw results returned from a query to an array of dictionaries
        :param res: results from a call to cur.fetchall()
        :param fields: optionally specify the fields if they are not the table fields
        :return: arr of dictionaries
        """
        if not len(res):
            return []
        fields = fields or self.fields
        if len(res[0]) != len(fields):
            raise AttributeError("Length of result vector (%d) does not match number of fields (%d)" % (
                len(res[0]), len(fields)))
        out = []
        for r in res:
            out.append(
                dict([(f, v) for f, v in zip(fields, r)])
            )
        return out

    def execute_many(self, query_arr):
        self.cur.execute('\n'.join(query_arr))

    def insert_query(self, **kwargs):
        return """
        INSERT INTO {0}({1}) VALUES({2});
        """.format(
            self.table_name,
            ','.join(kwargs.keys()),
            ','.join([str(x) if x is not None else 'NULL' for x in kwargs.values()]),
        )

    def insert(self, **kwargs):
        self.cur.execute(self.insert_query(**kwargs))

    def insert_many_query(self, records, fields=None):
        """
        Generate bulk insert query.  If fields are provided, use these as the schema, otherwise assume that the
        first record is representative
        :param records:
        :param fields:
        :return:
        """
        fields = fields or records[0].keys()
        fields_str = ','.join(fields)
        qry = """INSERT INTO {0}({1}) VALUES\n""".format(
            self.table_name,
            fields_str
        )
        n = len(records)
        for i, rec in enumerate(records):
            vals = [str(rec.get(k)) if rec.get(k) is not None else 'NULL' for k in fields]
            qry += """({0})""".format(','.join(vals))
            if i == (n - 1):
                qry += ";"
            else:
                qry += ",\n"
        return qry

    def insert_many(self, records):
        # records is an iterable containing dictionaries that generate queries
        self.cur.execute(self.insert_many_query(records))

    def execute_and_fetch(self, qry, fields=None, convert_to_dict=False):
        """
        Execute the supplied query and fetch all results .
        Optionally cast each result to a dictionary
        NB: Conversion incurs a speed and memory penalty of around 50%
        :param qry:
        :param fields: Optional list of fields - this is required if convert_to_dict==True and the query does not
        return all table fields
        :param convert_to_dict: If True, cast every row returned to a dictionary
        :return:
        """
        self.cur.execute(qry)
        if convert_to_dict:
            return self.results_to_dict(self.cur.fetchall(), fields=fields)
        else:
            return self.cur.fetchall()

    def get_many(self, limit=100):
        # retrieve all results, imposing the limit (set limit=None or 0 for all results)
        qry = 'SELECT * FROM {0}'.format(self.table_name)
        if limit:
            qry += ' LIMIT {0}'.format(limit)
        return self.execute_and_fetch(qry, convert_to_dict=True)

    def end_transaction_block(self):
        # useful if an error occurs - cursor will not allow any further transactions until broken block is ended
        self.cur.execute('END;')

    @property
    def count(self):
        self.cur.execute('SELECT COUNT(*) FROM {0};'.format(self.table_name))
        return self.cur.fetchone()[0]

    def update_query(self, set_qry, where_qry=None, from_str=None, returning=None):
        where_qry = where_qry or {}
        set_str = []
        for k, v in set_qry.items():
            if v is None:
                # set to NULL for SQL
                v = 'NULL'
            set_str.append("{0} = {1}".format(k, v))
        qry = """
        UPDATE {0} SET {1}
        """.format(
            self.table_name,
            ', '.join(set_str),
            )
        if from_str:
            qry += """
            FROM {0}
            """.format(from_str)
        if where_qry:
            qry += where_query_from_dict(where_qry)

        if returning is not None:
            qry += " RETURNING {0}".format(returning)

        qry += ';'

        return qry

    def update(self, set_qry, where_qry=None, from_str=None):
        self.cur.execute(self.update_query(set_qry, where_qry, from_str))

    def upsert_query(self, set_qry, where_qry):
        """
        Update record if it exists, else create it.
        Limitations: 1) where_qry must be a simple equality for the insert function to work
        2) FROM keyword not supported (yet)
        :param set_qry: Dictionary of {field: value} pairs for the SET portion of the query
        :param where_qry: Dictionary of {field: value} pairs for the WHERE portion of the query.  A value string
        commencing with '*' denotes an operator other than = (which must be included in the string)
        :return:
        """
        update_qry = self.update_query(set_qry, where_qry=where_qry, returning='*').replace(';', '')  # cut off final ;

        # insert query is a bit different from the usual format
        insert_qry = """
        INSERT INTO {0}({1}) SELECT {2}
        """.format(
            self.table_name,
            ','.join(set_qry.keys()),
            ','.join([str(x) if x is not None else 'NULL' for x in set_qry.values()]),
        )

        qry = """
        WITH upsert AS ({0}) {1} WHERE NOT EXISTS (SELECT * FROM upsert);
        """.format(
            update_qry,
            insert_qry
        )
        return qry

    def upsert(self, set_qry, where_qry):
        self.cur.execute(self.upsert_query(set_qry, where_qry))

    def select_query(self, where_dict=None, fields=None, limit=None):
        qry = """
        SELECT {0} FROM {1}
        """.format(
            ', '.join(fields) if fields else '*',
            self.table_name
        )
        if where_dict is not None:
            qry += where_query_from_dict(where_dict)
        if limit is not None:
            qry += """
            LIMIT {0}
            """.format(int(limit))
        qry += ';'
        return qry

    def select(self, where_dict=None, fields=None, limit=None, convert_to_dict=True):
        qry = self.select_query(where_dict=where_dict, fields=fields, limit=limit)
        return self.execute_and_fetch(qry, fields=fields, convert_to_dict=convert_to_dict)


class SanFrancisco(GeosTable):
    table_name = 'sanfrancisco'
    schema = (
        'id SERIAL PRIMARY KEY',
        'incident_number VARCHAR(9)',
        'datetime TIMESTAMP',
        'location GEOMETRY(POINT, 26943)',  # US NAD 83 zone 3
        'category VARCHAR(32) NOT NULL',
    )

    spatial_indices = {
        'gix': 'location'
    }


class LosAngeles(GeosTable):
    table_name = 'losangeles'
    schema = (
        'id SERIAL PRIMARY KEY',
        'incident_number VARCHAR(9)',
        'datetime TIMESTAMP',
        'location GEOMETRY(POINT, 26945)',  # US NAD 83 zone 5
        'category VARCHAR(64) NOT NULL',
    )

    spatial_indices = {
        'gix': 'location'
    }


class Chic(GeosTable):
    # TODO: replace Chicago table with this one
    table_name = 'chicago'
    schema = (
        'id INTEGER PRIMARY KEY',
        'case_number VARCHAR(16)',
        'datetime TIMESTAMP',
        'location GEOMETRY(POINT, 2028)',  # TODO: switch to 26916, US NAD 83 zone 16
        'primary_type VARCHAR(64)',
        'description VARCHAR(128)',
        'arrest_made BOOLEAN'
    )

    spatial_indices = {
        'gix': 'location'
    }


class Ocu(models.Model):
    code = models.CharField(help_text='OCU code', max_length=3, primary_key=True)
    description = models.CharField(help_text='OCU interpretation', max_length=128)

    def __str__(self):
        return "%s - %s" % (self.code, self.description)


class Nicl(models.Model):
    number = models.IntegerField(help_text='NICL category number', primary_key=True)
    group = models.CharField(help_text='NICL group', max_length=128)
    description = models.CharField(help_text='NICL interpretation', max_length=128)
    category_number = models.IntegerField(help_text='Top permissible NICL level', null=True, blank=True)
    category_letter = models.CharField(help_text='NICL classing', max_length=8, null=True, blank=True)

    def __str__(self):
        return "%03u - %s" % (self.number, self.description)


class Cad(models.Model):
    call_sign = models.CharField(help_text='Call sign', max_length=8, null=True, blank=True)
    res_type = models.CharField(help_text='Unknown', max_length=4, null=True, blank=True)
    res_own = models.ForeignKey('Ocu', help_text='Resource owner?', null=True, blank=True, related_name='res_own_set')
    inc_number = models.IntegerField(help_text='Incident number', null=False)
    inc_datetime = models.DateTimeField(help_text='Date and time of incident', null=False, blank=False)
    inc_weekday = models.CharField(help_text='Day of the week', max_length=3, choices=WEEKDAY_CHOICES, null=False, blank=False)
    op01 = models.ForeignKey('Nicl', help_text='Reported NICL classification 1', null=True, blank=True, related_name='op1_set')
    op02 = models.ForeignKey('Nicl', help_text='Reported NICL classification 2', null=True, blank=True, related_name='op2_set')
    op03 = models.ForeignKey('Nicl', help_text='Reported NICL classification 3', null=True, blank=True, related_name='op3_set')
    cl01 = models.ForeignKey('Nicl', help_text='Assigned NICL classification 1', null=True, blank=True, related_name='cl1_set')
    cl02 = models.ForeignKey('Nicl', help_text='Assigned NICL classification 2', null=True, blank=True, related_name='cl2_set')
    cl03 = models.ForeignKey('Nicl', help_text='Assigned NICL classification 3', null=True, blank=True, related_name='cl3_set')
    att_bocu = models.ForeignKey('Ocu', help_text='MPS OCU', null=True, blank=True, related_name='cad_set')
    caller = models.CharField(help_text='Type of person reporting incident', max_length=1, choices=CALLER_CHOICES, null=True, blank=True)
    cris_entry = models.CharField(help_text='CRIS entry if exists', max_length=16, null=True, blank=True) # TODO: reference table
    units_assigned_number = models.IntegerField(help_text='Incident number', null=False, blank=False, default=0)
    grade = models.CharField(help_text='Operator grading of incident', max_length=1, choices=GRADE_CHOICES, null=True, blank=True, default=None)
    uc = models.BooleanField(help_text='Unknown', default=False)
    arrival_datetime = models.DateTimeField(help_text='Date and time of arrival', null=True, blank=True)
    response_time = models.DecimalField(help_text='Response time in seconds', max_digits=10, decimal_places=2, null=True, blank=True)
    att_map = models.PointField(help_text='Location from resources GPS system', srid=27700, null=True, blank=True, default=None)
    inc_map = models.PointField(help_text='Location of incident', srid=27700, null=True, blank=True, default=None)
    call_map = models.PointField(help_text='Location of caller', srid=27700, null=True, blank=True, default=None)

    objects = models.GeoManager()

    def __str__(self):
        return "%u - %s - %s" % (self.inc_number, self.call_sign, str(self.inc_datetime))

    class Meta:
        app_label = 'database'


class Chicago(models.Model):
    number = models.IntegerField(help_text='CPD crime number', primary_key=True)
    case_number = models.CharField(help_text='CPD case number', max_length=16)
    datetime = models.DateTimeField(help_text='Date and time of incident')
    block = models.CharField(help_text='Location block descriptor', max_length=64)
    iucr = models.CharField(help_text='IUCR crime code', max_length=8)
    primary_type = models.CharField(help_text='Primary crime type', max_length=64)
    description = models.CharField(help_text='Description of crime', max_length=128)
    location_type = models.CharField(help_text='Nature of crime location', max_length=64)
    arrest = models.BooleanField(help_text='Was an arrest made?', default=False)
    domestic = models.BooleanField(help_text='Is incident domestic?', default=False)
    location = models.PointField(help_text='Crime location', srid=2028)

    objects = models.GeoManager()

    def __str__(self):
        return "%d - %s - %s" % (self.number, str(self.datetime), self.primary_type)

class DivisionType(models.Model):
    name = models.CharField(help_text='Name for this division set', max_length=128, primary_key=True)
    description = models.TextField(help_text='Text description of this division set')

    def __str__(self):
        return self.name


class Division(models.Model):
    name = models.CharField(help_text='Region name', max_length=50)
    code = models.CharField(help_text='Region code', max_length=50, unique=True, null=True, blank=True)
    type = models.ForeignKey('DivisionType', help_text='Type of division', related_name='division_set')
    mpoly = models.MultiPolygonField(srid=27700)

    objects = models.GeoManager()

    def __str__(self):
        if self.type:
            return "%s - %s" % (self.type.name, self.name)
        else:
            return "None - %s" % self.name

    class Meta:
        unique_together = ('name', 'type', 'code')


class ChicagoDivision(models.Model):
    name = models.CharField(help_text='Region name', max_length=50)
    type = models.ForeignKey('DivisionType', help_text='Type of division', related_name='chicagodivision_set')
    mpoly = models.MultiPolygonField(srid=2028)

    objects = models.GeoManager()

    def __str__(self):
        if self.type:
            return "%s - %s" % (self.type.name, self.name)
        else:
            return "None - %s" % self.name

    class Meta:
        unique_together = ('name', 'type')


class Cris(models.Model):
    date = models.DateField(help_text='First day of month')
    lsoa_code = models.CharField(help_text='LSOA code', max_length=16)
    lsoa = models.ForeignKey('Division', to_field='code', help_text='LSOA', null=True, blank=True, on_delete=models.SET_NULL)
    crime_major = models.CharField(help_text='Major text for crime type', max_length=128)
    crime_minor = models.CharField(help_text='Minor text for crime type', max_length=128)
    count = models.IntegerField(help_text='Crime count', default=0)

    def __str__(self):
        return '%s - %s - %s' % (self.lsoa_code, str(self.date), self.crime_minor)

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        # check correct lsoa link
        if self.lsoa:
            if self.lsoa.type.name != "lsoa":
                raise IntegrityError("CRIS entry linked to division that is not an LSOA")
        super(Cris, self).save(force_insert, force_update, using, update_fields)

