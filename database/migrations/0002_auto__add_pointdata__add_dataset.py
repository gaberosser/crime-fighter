# -*- coding: utf-8 -*-
from south.utils import datetime_utils as datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Adding model 'PointData'
        db.create_table(u'database_pointdata', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('location', self.gf('django.contrib.gis.db.models.fields.PointField')(default=None, srid=27700, null=True, blank=True)),
            ('time', self.gf('django.db.models.fields.DateTimeField')()),
            ('spatial_uncertainty', self.gf('django.db.models.fields.FloatField')(default=None, null=True, blank=True)),
            ('temporal_uncertainty', self.gf('django.db.models.fields.FloatField')(default=None, null=True, blank=True)),
            ('dataset', self.gf('django.db.models.fields.related.ForeignKey')(related_name='dataset', to=orm['database.Dataset'])),
        ))
        db.send_create_signal(u'database', ['PointData'])

        # Adding model 'Dataset'
        db.create_table(u'database_dataset', (
            ('name', self.gf('django.db.models.fields.CharField')(max_length=32, primary_key=True)),
            ('description', self.gf('django.db.models.fields.CharField')(max_length=256, null=True, blank=True)),
            ('region', self.gf('django.contrib.gis.db.models.fields.MultiPolygonField')(srid=27700, null=True, blank=True)),
        ))
        db.send_create_signal(u'database', ['Dataset'])


    def backwards(self, orm):
        # Deleting model 'PointData'
        db.delete_table(u'database_pointdata')

        # Deleting model 'Dataset'
        db.delete_table(u'database_dataset')


    models = {
        'database.cad': {
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
        u'database.cris': {
            'Meta': {'object_name': 'Cris'},
            'count': ('django.db.models.fields.IntegerField', [], {'default': '0'}),
            'crime_major': ('django.db.models.fields.CharField', [], {'max_length': '128'}),
            'crime_minor': ('django.db.models.fields.CharField', [], {'max_length': '128'}),
            'date': ('django.db.models.fields.DateField', [], {}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'lsoa': ('django.db.models.fields.related.ForeignKey', [], {'to': u"orm['database.Division']", 'to_field': "'code'", 'null': 'True', 'on_delete': 'models.SET_NULL', 'blank': 'True'}),
            'lsoa_code': ('django.db.models.fields.CharField', [], {'max_length': '16'})
        },
        u'database.dataset': {
            'Meta': {'object_name': 'Dataset'},
            'description': ('django.db.models.fields.CharField', [], {'max_length': '256', 'null': 'True', 'blank': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '32', 'primary_key': 'True'}),
            'region': ('django.contrib.gis.db.models.fields.MultiPolygonField', [], {'srid': '27700', 'null': 'True', 'blank': 'True'})
        },
        u'database.division': {
            'Meta': {'unique_together': "(('name', 'type', 'code'),)", 'object_name': 'Division'},
            'code': ('django.db.models.fields.CharField', [], {'max_length': '50', 'unique': 'True', 'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'mpoly': ('django.contrib.gis.db.models.fields.MultiPolygonField', [], {'srid': '27700'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '50'}),
            'type': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'division_set'", 'to': u"orm['database.DivisionType']"})
        },
        u'database.divisiontype': {
            'Meta': {'object_name': 'DivisionType'},
            'description': ('django.db.models.fields.TextField', [], {}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '128', 'primary_key': 'True'})
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
        },
        u'database.pointdata': {
            'Meta': {'object_name': 'PointData'},
            'dataset': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'dataset'", 'to': u"orm['database.Dataset']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'location': ('django.contrib.gis.db.models.fields.PointField', [], {'default': 'None', 'srid': '27700', 'null': 'True', 'blank': 'True'}),
            'spatial_uncertainty': ('django.db.models.fields.FloatField', [], {'default': 'None', 'null': 'True', 'blank': 'True'}),
            'temporal_uncertainty': ('django.db.models.fields.FloatField', [], {'default': 'None', 'null': 'True', 'blank': 'True'}),
            'time': ('django.db.models.fields.DateTimeField', [], {})
        }
    }

    complete_apps = ['database']