from django.contrib.gis.db import models
from django.db.models.signals import pre_save
from django.db import IntegrityError

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

# Create your models here.
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
    # TODO: x_coord and y_coord are in projection 102671 (NAD 1983 StatePlane Illinois East FIPS 1201 Feet)
    # convert location to that srid or 2028 (UTM zone 16N) instead for faster retrieval
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
    location = models.PointField(help_text='Lat/long of crime location', srid=4326)  ## TODO: switch to 2028, repopulate
    x_coord = models.IntegerField(help_text='x coord, system unknown') ## TODO: remove
    y_coord = models.IntegerField(help_text='y coord, system unknown')

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

