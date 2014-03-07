# -*- coding: utf-8 -*-
from south.utils import datetime_utils as datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Adding model 'Ocu'
        db.create_table(u'database_ocu', (
            ('code', self.gf('django.db.models.fields.CharField')(max_length=3, primary_key=True)),
            ('description', self.gf('django.db.models.fields.CharField')(max_length=128)),
        ))
        db.send_create_signal(u'database', ['Ocu'])

        # Adding model 'Nicl'
        db.create_table(u'database_nicl', (
            ('number', self.gf('django.db.models.fields.IntegerField')(primary_key=True)),
            ('group', self.gf('django.db.models.fields.CharField')(max_length=128)),
            ('description', self.gf('django.db.models.fields.CharField')(max_length=128)),
            ('category_number', self.gf('django.db.models.fields.IntegerField')(null=True, blank=True)),
            ('category_letter', self.gf('django.db.models.fields.CharField')(max_length=8, null=True, blank=True)),
        ))
        db.send_create_signal(u'database', ['Nicl'])

        # Adding model 'Cad'
        db.create_table(u'database_cad', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('call_sign', self.gf('django.db.models.fields.CharField')(max_length=8, null=True, blank=True)),
            ('res_type', self.gf('django.db.models.fields.CharField')(max_length=4, null=True, blank=True)),
            ('res_own', self.gf('django.db.models.fields.related.ForeignKey')(blank=True, related_name='res_own_set', null=True, to=orm['database.Ocu'])),
            ('inc_number', self.gf('django.db.models.fields.IntegerField')()),
            ('inc_datetime', self.gf('django.db.models.fields.DateTimeField')()),
            ('inc_weekday', self.gf('django.db.models.fields.CharField')(max_length=3)),
            ('op01', self.gf('django.db.models.fields.related.ForeignKey')(blank=True, related_name='op1_set', null=True, to=orm['database.Nicl'])),
            ('op02', self.gf('django.db.models.fields.related.ForeignKey')(blank=True, related_name='op2_set', null=True, to=orm['database.Nicl'])),
            ('op03', self.gf('django.db.models.fields.related.ForeignKey')(blank=True, related_name='op3_set', null=True, to=orm['database.Nicl'])),
            ('cl01', self.gf('django.db.models.fields.related.ForeignKey')(blank=True, related_name='cl1_set', null=True, to=orm['database.Nicl'])),
            ('cl02', self.gf('django.db.models.fields.related.ForeignKey')(blank=True, related_name='cl2_set', null=True, to=orm['database.Nicl'])),
            ('cl03', self.gf('django.db.models.fields.related.ForeignKey')(blank=True, related_name='cl3_set', null=True, to=orm['database.Nicl'])),
            ('att_bocu', self.gf('django.db.models.fields.related.ForeignKey')(blank=True, related_name='cad_set', null=True, to=orm['database.Ocu'])),
            ('caller', self.gf('django.db.models.fields.CharField')(max_length=1, null=True, blank=True)),
            ('cris_entry', self.gf('django.db.models.fields.CharField')(max_length=16, null=True, blank=True)),
            ('units_assigned_number', self.gf('django.db.models.fields.IntegerField')(default=0)),
            ('grade', self.gf('django.db.models.fields.CharField')(default=None, max_length=1, null=True, blank=True)),
            ('uc', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('arrival_datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
            ('response_time', self.gf('django.db.models.fields.DecimalField')(null=True, max_digits=10, decimal_places=2, blank=True)),
            ('att_map', self.gf('django.contrib.gis.db.models.fields.PointField')(default=None, srid=27700, null=True, blank=True)),
            ('inc_map', self.gf('django.contrib.gis.db.models.fields.PointField')(default=None, srid=27700, null=True, blank=True)),
            ('call_map', self.gf('django.contrib.gis.db.models.fields.PointField')(default=None, srid=27700, null=True, blank=True)),
        ))
        db.send_create_signal(u'database', ['Cad'])

        # Adding model 'Borough'
        db.create_table(u'database_borough', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(unique=True, max_length=50)),
            ('area', self.gf('django.db.models.fields.FloatField')()),
            ('nonld_area', self.gf('django.db.models.fields.FloatField')(default=0.0)),
            ('mpoly', self.gf('django.contrib.gis.db.models.fields.MultiPolygonField')(srid=27700)),
        ))
        db.send_create_signal(u'database', ['Borough'])


    def backwards(self, orm):
        # Deleting model 'Ocu'
        db.delete_table(u'database_ocu')

        # Deleting model 'Nicl'
        db.delete_table(u'database_nicl')

        # Deleting model 'Cad'
        db.delete_table(u'database_cad')

        # Deleting model 'Borough'
        db.delete_table(u'database_borough')


    models = {
        u'database.borough': {
            'Meta': {'object_name': 'Borough'},
            'area': ('django.db.models.fields.FloatField', [], {}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'mpoly': ('django.contrib.gis.db.models.fields.MultiPolygonField', [], {'srid': '27700'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '50'}),
            'nonld_area': ('django.db.models.fields.FloatField', [], {'default': '0.0'})
        },
        u'database.cad': {
            'Meta': {'object_name': 'Cad'},
            'arrival_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'att_bocu': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'cad_set'", 'null': 'True', 'to': u"orm['database.Ocu']"}),
            'att_map': ('django.contrib.gis.db.models.fields.PointField', [], {'default': 'None', 'srid': '27700', 'null': 'True', 'blank': 'True'}),
            'call_map': ('django.contrib.gis.db.models.fields.PointField', [], {'default': 'None', 'srid': '27700', 'null': 'True', 'blank': 'True'}),
            'call_sign': ('django.db.models.fields.CharField', [], {'max_length': '8', 'null': 'True', 'blank': 'True'}),
            'caller': ('django.db.models.fields.CharField', [], {'max_length': '1', 'null': 'True', 'blank': 'True'}),
            'cl01': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'cl1_set'", 'null': 'True', 'to': u"orm['database.Nicl']"}),
            'cl02': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'cl2_set'", 'null': 'True', 'to': u"orm['database.Nicl']"}),
            'cl03': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'cl3_set'", 'null': 'True', 'to': u"orm['database.Nicl']"}),
            'cris_entry': ('django.db.models.fields.CharField', [], {'max_length': '16', 'null': 'True', 'blank': 'True'}),
            'grade': ('django.db.models.fields.CharField', [], {'default': 'None', 'max_length': '1', 'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'inc_datetime': ('django.db.models.fields.DateTimeField', [], {}),
            'inc_map': ('django.contrib.gis.db.models.fields.PointField', [], {'default': 'None', 'srid': '27700', 'null': 'True', 'blank': 'True'}),
            'inc_number': ('django.db.models.fields.IntegerField', [], {}),
            'inc_weekday': ('django.db.models.fields.CharField', [], {'max_length': '3'}),
            'op01': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'op1_set'", 'null': 'True', 'to': u"orm['database.Nicl']"}),
            'op02': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'op2_set'", 'null': 'True', 'to': u"orm['database.Nicl']"}),
            'op03': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'op3_set'", 'null': 'True', 'to': u"orm['database.Nicl']"}),
            'res_own': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'res_own_set'", 'null': 'True', 'to': u"orm['database.Ocu']"}),
            'res_type': ('django.db.models.fields.CharField', [], {'max_length': '4', 'null': 'True', 'blank': 'True'}),
            'response_time': ('django.db.models.fields.DecimalField', [], {'null': 'True', 'max_digits': '10', 'decimal_places': '2', 'blank': 'True'}),
            'uc': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'units_assigned_number': ('django.db.models.fields.IntegerField', [], {'default': '0'})
        },
        u'database.nicl': {
            'Meta': {'object_name': 'Nicl'},
            'category_letter': ('django.db.models.fields.CharField', [], {'max_length': '8', 'null': 'True', 'blank': 'True'}),
            'category_number': ('django.db.models.fields.IntegerField', [], {'null': 'True', 'blank': 'True'}),
            'description': ('django.db.models.fields.CharField', [], {'max_length': '128'}),
            'group': ('django.db.models.fields.CharField', [], {'max_length': '128'}),
            'number': ('django.db.models.fields.IntegerField', [], {'primary_key': 'True'})
        },
        u'database.ocu': {
            'Meta': {'object_name': 'Ocu'},
            'code': ('django.db.models.fields.CharField', [], {'max_length': '3', 'primary_key': 'True'}),
            'description': ('django.db.models.fields.CharField', [], {'max_length': '128'})
        }
    }

    complete_apps = ['database']