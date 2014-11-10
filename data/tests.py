__author__ = 'gabriel'
from django.test import SimpleTestCase
import models
import numpy as np
import ipdb


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
        # 3D
        x = np.meshgrid(np.linspace(0, 1, 10), np.linspace(1, 2, 10), np.linspace(2, 3, 10))
        data = models.DataArray(np.concatenate([t[..., np.newaxis] for t in x], axis=3))
        self.assertTrue(isinstance(data, models.Data))
        self.assertEqual(data.nd, 3)
        self.assertEqual(data.ndata, 1000)
        self.assertTrue(np.all(data[:, 2] == x[2].flatten('F')))

        # 5D
        x = np.linspace(0, 1, 500).reshape(10, 10, 5)
        data = models.DataArray(x)
        self.assertTrue(isinstance(data, models.Data))
        self.assertEqual(data.nd, 5)
        self.assertEqual(data.ndata, 100)
        self.assertTrue(np.all(data[:, 3] == x[..., 3].flatten('F')))

    def test_separate(self):

        # 2D, no original shape
        x = np.random.rand(50, 2)
        data = models.DataArray(x)
        self.assertTrue(data.original_shape is None)
        a, b = data.separate
        # a and b should be an np.ndarray instance
        self.assertIsInstance(a, np.ndarray)
        # ... and NOT a datatype any more
        self.assertFalse(isinstance(a, models.Data))
        self.assertTrue(np.all(a == x[:, 0]))
        self.assertTrue(np.all(b == x[:, 1]))

        # 5D with original shape
        x = np.linspace(0, 1, 500).reshape(10, 10, 5)
        data = models.DataArray(x)
        self.assertTupleEqual(data.original_shape, (10, 10))
        res = data.separate
        self.assertEqual(len(res), 5)
        for i in range(5):
            self.assertTupleEqual(res[i].shape, (10, 10))
            self.assertTrue(np.all(res[i] == x[:, :, i]))


        x = np.meshgrid(np.linspace(0, 1, 10), np.linspace(1, 2, 10), np.linspace(2, 3, 10))
        data = models.DataArray(np.concatenate([t[..., np.newaxis] for t in x], axis=3))
        self.assertTupleEqual(data.original_shape, (10, 10, 10))
        res = data.separate
        for i in range(3):
            self.assertTrue(np.all(res[i] == x[i]))
            self.assertIsInstance(res[i], np.ndarray)
            self.assertFalse(isinstance(res[i], models.Data))





class TestSpaceTimeDataArray(SimpleTestCase):

    def test_exceptions(self):
        with self.assertRaises(AttributeError):
            data = models.SpaceTimeDataArray([1, 2, 3])

    def test_instantiation(self):
        data = models.SpaceTimeDataArray([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(isinstance(data, models.Data))
        self.assertEqual(data.nd, 2)
        self.assertEqual(data.ndata, 3)

        self.assertTrue(isinstance(data.space, models.DataArray))
        self.assertTrue(np.all(data[:, 1] == data.space[:, 0]))
        self.assertTrue(np.all(data[:, 1] == np.array([2, 4, 6])))

        x = np.linspace(0, 1, 500).reshape(10, 10, 5)
        data = models.SpaceTimeDataArray(x)
        self.assertEqual(data.nd, 5)
        self.assertEqual(data.ndata, 100)

        self.assertTrue(isinstance(data.space, models.DataArray))

    def test_slicing_etc(self):
        data = models.SpaceTimeDataArray([[1, 2], [3, 4], [5, 6]])

        self.assertTrue(data.getdim(0) == data.time)
        self.assertTrue(np.all(data[:, 0] == np.array([1, 3, 5])))

        x = np.linspace(0, 1, 500).reshape(10, 10, 5)
        data = models.SpaceTimeDataArray(x)

        self.assertTrue(data.time == models.DataArray(x[..., 0].flatten('F')))

        self.assertTrue(np.all(data.space[:, 0] == x[..., 1].flatten('F')))
        self.assertTrue(np.all(data.space[:, 3] == x[..., 4].flatten('F')))

    def test_set_get(self):

        data = models.SpaceTimeDataArray([[1, 2], [3, 4], [5, 6]])
        # get time /space
        self.assertTrue(np.all(data.getdim(0) == data.time))
        self.assertTrue(np.all(data.getdim(1) == data.space))
        t = models.DataArray([3, 6, 9])
        data.time = t
        self.assertTrue(data.getdim(0) == t)
        s = models.DataArray([4, 8, 12])
        data.space = s
        self.assertTrue(data.getdim(1) == s)






class TestCartesianData(SimpleTestCase):
    pass


