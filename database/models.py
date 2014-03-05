from django.contrib.gis.db import models

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
    att_map = models.PointField(help_text='Location from resources GPS system', srid=32631, null=True, blank=True, default=None)
    inc_map = models.PointField(help_text='Location of incident', srid=32631, null=True, blank=True, default=None)
    call_map = models.PointField(help_text='Location of caller', srid=32631, null=True, blank=True, default=None)

    def __str__(self):
        return "%u - %s - %s" % (self.inc_number, self.call_sign, str(self.inc_datetime))