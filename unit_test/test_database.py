__author__ = 'gabriel'
from database import sandbox
from test_data import csv_data
import unittest
import csv

class TestNiclTable(unittest.TestCase):

    def setUp(self):
        self.nicl = sandbox.Nicl()

    def test_creation(self):
        self.assertEqual(self.nicl.conn.closed, 0)
        cur = self.nicl.conn.cursor()
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s;", (self.nicl.name, ))
        res = cur.fetchall()
        self.assertTrue(res)

    def test_populate(self):
        c = csv.reader(csv_data.nicl_csv)
        self.nicl.populate(c)
        cur = self.nicl.conn.cursor()
        cur.execute("SELECT * FROM %s;" % self.nicl.name)
        res = cur.fetchall()
        self.assertEqual(len(res), 4)
        self.assertTupleEqual(res[0], (1, 'Test category A', 'DescA', 1, 'A'))
        self.assertTupleEqual(res[3], (4, 'Test category B', 'DescD', 2, 'B'))

class TestOcuTable(unittest.TestCase):

    def setUp(self):
        self.ocu = sandbox.Ocu()

    def test_creation(self):
        self.assertEqual(self.ocu.conn.closed, 0)
        cur = self.ocu.conn.cursor()
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s;", (self.ocu.name, ))
        res = cur.fetchall()
        self.assertTrue(res)

    def test_populate(self):
        c = csv.reader(csv_data.ocu_csv)
        self.ocu.populate(c)
        cur = self.ocu.conn.cursor()
        cur.execute("SELECT * FROM %s;" % self.ocu.name)
        res = cur.fetchall()
        self.assertEqual(len(res), 3)
        self.assertTupleEqual(res[0], ('AA', 'catAA'))
        self.assertTupleEqual(res[2], ('AC', 'catAC'))