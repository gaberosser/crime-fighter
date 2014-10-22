__author__ = 'gabriel'
from django.test import SimpleTestCase
import models
import numpy as np


class TestDataArray(SimpleTestCase):

    def test_instantiation_1d(self):
        data = models.DataArray([1])
        self.assertTrue(isinstance(data, models.Data))
        self.assertEqual(data.nd, 1)
        self.assertEqual(data.ndata, 1)

        data = models.DataArray([1, 2, 3])
        self.assertEqual(data.nd, 1)
        self.assertEqual(data.ndata, 3)

        data = models.DataArray(np.arange(5))
        self.assertEqual(data.nd, 1)
        self.assertEqual(data.ndata, 5)

        data = models.DataArray(np.arange(5).reshape(5, 1))
        self.assertEqual(data.nd, 1)
        self.assertEqual(data.ndata, 5)

    def test_instantiation_2d(self):
        data = models.DataArray([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(isinstance(data, models.Data))
        self.assertEqual(data.nd, 2)
        self.assertEqual(data.ndata, 3)
        self.assertTrue(np.all(data[:, 0] == np.array([1, 3, 5])))
        self.assertTrue(np.all(data[1] == np.array([3, 4])))

        x = np.random.rand(50, 2)
        data = models.DataArray(x)
        self.assertEqual(data.nd, 2)
        self.assertEqual(data.ndata, 50)
        self.assertTrue(np.all(data[:, 1] == x[:, 1]))

    def test_instantiation_nd(self):
        x = np.linspace(0, 1, 500).reshape(10, 10, 5)
        data = models.DataArray(x)
        self.assertTrue(isinstance(data, models.Data))
        self.assertEqual(data.nd, 5)
        self.assertEqual(data.ndata, 100)
        self.assertTrue(np.all(data[:, 3] == x[..., 3].flatten('F')))


class TestSpaceTimeDataArray(SimpleTestCase):

    def test_exceptions(self):
        with self.assertRaises(AttributeError):
            data = models.SpaceTimeDataArray([1, 2, 3])

    def test_data_class(self):
        data = models.SpaceTimeDataArray([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(isinstance(data, models.Data))
        self.assertEqual(data.nd, 2)
        self.assertEqual(data.ndata, 3)
        self.assertTrue(np.all(data[:, 0] == data.time))
        self.assertTrue(np.all(data[:, 0] == np.array([1, 3, 5])))

        self.assertTrue(isinstance(data.space, models.DataArray))
        self.assertTrue(np.all(data[:, 1] == data.space[:, 0]))
        self.assertTrue(np.all(data[:, 1] == np.array([2, 4, 6])))

        x = np.linspace(0, 1, 500).reshape(10, 10, 5)
        data = models.SpaceTimeDataArray(x)
        self.assertEqual(data.nd, 5)
        self.assertEqual(data.ndata, 100)
        self.assertTrue(np.all(data.time == x[..., 0].flatten('F')))

        self.assertTrue(isinstance(data.space, models.DataArray))
        self.assertTrue(np.all(data.space[:, 0] == x[..., 1].flatten('F')))
        self.assertTrue(np.all(data.space[:, 3] == x[..., 4].flatten('F')))




class TestCartesianData(SimpleTestCase):
    pass


