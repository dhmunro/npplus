import unittest
import numpy as np

import npplus as npp
import npplus.basic as yl

import __main__

# nosetests --with-coverage --cover-package=npplus

def nearly_eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.shape == b.shape) and np.allclose(a, b)

class TestYorLike(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_span(self):
        """Check span function."""
        x = yl.span(-2., 5., 8)
        self.assertTrue(nearly_eq(np.arange(-2.,6.), x), "span failed 1")
        y = yl.span(5., -2., 8)
        self.assertTrue(nearly_eq(np.arange(5.,-3.,-1), y), "span failed 2")
        xy = np.array([x, y])
        p = yl.span([-2.,5.], [5.,-2.], 8)
        self.assertTrue(nearly_eq(xy.T, p), "span failed 3")
        p = yl.span([-2.,5.], [5.,-2.], 8, axis=-1)
        self.assertTrue(nearly_eq(xy, p), "span failed 4")

    def test_spanl(self):
        """Check spanl function."""
        seq = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
        x = yl.spanl(0.25, 8., 6)
        self.assertTrue(nearly_eq(seq, x), "spanl failed 1")
        y = yl.spanl(-8., -0.25, 6)
        self.assertTrue(nearly_eq(-seq[::-1], y), "spanl failed 2")
        xy = np.array([x, y])
        p = yl.spanl([0.25, -8.], [8.,-.25], 6, axis=-1)
        self.assertTrue(nearly_eq(xy, p), "spanl failed 3")
        with self.assertRaises(ValueError) as cm:
            y = yl.spanl(-8., 0.25, 6)
        self.assertIsInstance(cm.exception, ValueError, "spanl failed 4")

    def test_cat(self):
        """Check cat_ function."""
        one = np.arange(4)
        two = np.arange(10,130,10).reshape(3,4)
        x = yl.cat_(one, 0, two)
        self.assertEqual((5,4), x.shape, "cat_ failed shape 1")
        self.assertTrue(np.array_equal(x[0], one) and np.array_equiv(x[1], 0)
                        and np.array_equal(x[2:], two), "cat_ failed value 1")
        x = yl.cat_(one, 0, two, axis=1)
        self.assertEqual((3,9), x.shape, "cat_ failed shape 2")
        self.assertTrue(np.array_equiv(x[:,:4], one) and
                        np.array_equiv(x[:,4:5], 0) and
                        np.array_equal(x[:,5:], two), "cat_ failed value 2")
        with self.assertRaises(TypeError):
            y = yl.cat_(one, 0, two, xaxis=1)
        with self.assertRaises(ValueError):
            y = yl.cat_(one, 0, two, axis=2)

    def test_a(self):
        """Check a_ function."""
        one = np.arange(4)
        two = np.arange(10,130,10).reshape(3,4)
        x = yl.a_(one, 0, two)
        self.assertEqual((3,3,4), x.shape, "a_ failed shape 1")
        self.assertTrue(np.array_equiv(x[0], one) and np.array_equiv(x[1], 0)
                        and np.array_equal(x[2], two), "a_ failed value 1")
        x = yl.a_(one, 0, two, axis=-1)
        self.assertEqual((3,4,3), x.shape, "a_ failed shape 2")
        self.assertTrue(np.array_equiv(x[:,:,0], one) and
                        np.array_equiv(x[:,:,1], 0) and
                        np.array_equal(x[:,:,2], two), "a_ failed value 2")
        with self.assertRaises(TypeError):
            y = yl.a_(one, 0, two, xaxis=1)
        with self.assertRaises(ValueError):
            y = yl.a_(one, 0, two, axis=3)

    def test_minmax(self):
        """Check max_ and min_ functions."""
        mn = yl.min_(3, np.arange(3), np.eye(3), -1)
        mx = yl.max_(3, np.arange(3), np.eye(3), -1)
        self.assertEqual((3,3), mn.shape, "min_ failed shape")
        self.assertTrue((mn == -1).all(), "min_ failed value")
        self.assertEqual((3,3), mx.shape, "max_ failed shape")
        self.assertTrue((mx == 3).all(), "max_ failed value")

    def test_abs(self):
        """Check abs_ function."""
        v = np.arange(-3.,9.).reshape(4,3)
        x = yl.abs_(v)
        self.assertTrue((np.absolute(v) == x).all(), "abs_ single failed")
        w = np.arange(3.)
        z = np.sqrt(v*v + 2.*w*w)
        x = yl.abs_(w, v, w)
        self.assertTrue(nearly_eq(z, x), "abs_ multiple failed")

    def test_atan(self):
        """Check atan function."""
        ad = np.r_[-158.:142.:16j] * np.pi/180.
        x, y = np.cos(ad), np.sin(ad)
        t = y/x
        z = yl.atan(t)
        self.assertTrue(nearly_eq(np.arctan(t), z), "atan single failed")
        z = yl.atan(y, x)
        self.assertTrue(nearly_eq(ad, z), "atan double failed 1")
        out = z.copy()
        z = yl.atan(y, x, branch=0., out=out)
        a = np.where(ad<0., 2.*np.pi+ad, ad)
        self.assertTrue(nearly_eq(a, z), "atan double failed 2")

    def test_cum(self):
        """Check cum function."""
        x = np.array([[1], [2]]).repeat(4, axis=1)
        y, r = yl.cum(x,axis=None), [0, 1, 2, 3, 4, 6, 8, 10, 12]
        self.assertTrue(np.array_equal(r, y), "cum failed 1")
        y, r = yl.cum(x, axis=0), np.array([[0], [1], [3]]).repeat(4,axis=1)
        self.assertTrue(np.array_equal(r, y), "cum failed 2")
        y, r = yl.cum(x), np.array([np.arange(5), np.arange(0,10,2)])
        self.assertTrue(np.array_equal(r, y), "cum failed 3")
        with self.assertRaises(TypeError):
            y = yl.cum(20)

    def test_zcen(self):
        """Check zcen function."""
        x = np.arange(1,9).reshape(2, 4)
        y, r = yl.zcen(x), 0.5*(x[...,1:] + x[...,:-1])
        self.assertTrue(nearly_eq(r, y), "zcen failed 1")
        y, r = yl.zcen(x,axis=0), 0.5*(x[1:,...] + x[:-1,...])
        self.assertTrue(nearly_eq(r, y), "zcen failed 2")
        with self.assertRaises(TypeError):
            y = yl.zcen(20)
        with self.assertRaises(TypeError):
            y = yl.zcen([20])

    def test_pcen(self):
        """Check pcen function."""
        x = np.arange(1,9).reshape(2, 4)
        y, r = yl.pcen(x), np.array([[1,1.5,2.5,3.5,4],[5,5.5,6.5,7.5,8]])
        self.assertTrue(nearly_eq(r, y), "pcen failed 1")
        y, r = yl.pcen(x,axis=0), np.arange(1.,5.)+np.arange(0,6,2).reshape(3,1)
        self.assertTrue(nearly_eq(r, y), "pcen failed 2")
        y, r = yl.pcen(x[0:1,:],axis=0), x[0:1,:].repeat(2,axis=0)
        self.assertTrue(nearly_eq(r, y), "pcen failed 3")
        with self.assertRaises(TypeError):
            y = yl.pcen(20)

############################################################################

if __name__ == '__main__':
    unittest.main()
