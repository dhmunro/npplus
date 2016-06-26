import unittest
import numpy as np

import npplus as npp
import npplus.solveper as nppsp
import npplus.pwpoly as nppw

import __main__

# nosetests --with-coverage --cover-package=npplus

class TestSolvePeriodic(unittest.TestCase):
    def setUp(self):
        self.ab = np.r_[0.01:0.09:0.01] + np.r_[0.4:0.8:0.1][:,np.newaxis]
        self.lau = (2,1)
        self.ad = nppsp._diag_to_norm(self.lau, self.ab)
        a = self.ad.copy()
        a[-1:,:1] = nppsp._diag_to_norm((0,0), self.ab[:1,:1])
        a[:2,-2:] = nppsp._diag_to_norm((0,1), self.ab[-2:,-2:])
        self.a = a
        self.x = np.arange(8.)[:,np.newaxis].repeat(2,axis=-1)
        self.x[:,1] = self.x[::-1,0]
        self.b = a.dot(self.x)
        self.abl = self.ab[1:,:].copy()
        self.abu = self.ab[-1:0:-1,:].copy()
        self.abu[0] = np.roll(self.abu[0], 2)
        self.abu[1] = np.roll(self.abu[1], 1)
        ab = np.concatenate((self.abu[:-1], self.abl), axis=0)
        ad = nppsp._diag_to_norm((2,2), ab)
        a = ad.copy()
        a[-2:,:2] = nppsp._diag_to_norm((1,0), ab[:2,:2])
        a[:2,-2:] = nppsp._diag_to_norm((0,1), ab[-2:,-2:])
        self.aa = a
        self.bb = a.dot(self.x)

    def tearDown(self):
        pass

    def test_solve(self):
        """Check solve_periodic."""
        x = nppsp.solve_periodic(self.lau, self.ab, self.b)
        self.assertTrue(np.allclose(self.x, x), "solve_periodic failed 1")
        x = nppsp.solve_periodic(self.lau, self.ab, self.b[:,1])
        self.assertTrue(np.allclose(self.x[:,1], x), "solve_periodic failed 2")

    def test_solves(self):
        """Check solves_periodic."""
        xs = self.x
        x = nppsp.solves_periodic(self.abu, self.bb)
        self.assertTrue(np.allclose(xs, x), "solves_periodic failed 1")
        x = nppsp.solves_periodic(self.abu, self.bb[:,1])
        self.assertTrue(np.allclose(xs[:,1], x), "solves_periodic l failed 2")
        x = nppsp.solves_periodic(self.abl, self.bb, lower=True)
        self.assertTrue(np.allclose(xs, x), "solves_periodic failed 1")
        x = nppsp.solves_periodic(self.abl, self.bb[:,1], lower=True)
        self.assertTrue(np.allclose(xs[:,1], x), "solves_periodic l failed 2")

class TestPwPoly(unittest.TestCase):
    def setUp(self):
        self.xk = np.array([-1., 0., 2., 3.])
        self.yk = np.array([1., 1., -3., 5.])
        # here is a natural cubic spline on xk
        c0 = [1., 1., 0., 0.]
        c1 = [1., 1., 0., -1.]
        c2 = [1., -2., -3., 1.5]
        c3 = [-3., 4., 6., -2.]
        c4 = [ 5., 10., 0., 0.]
        self.c = np.array([c0, c1, c2, c3, c4]).T.copy()
        self.pw3 = nppw.PwPoly.new(self.xk, self.c)
        # as periodic ignoring yk[-1]
        c1 = [1., 3., -3.6, 0.6]
        c2 = [1., -2.4, -1.8, 1.]
        c3 = [-3., 2.4, 4.2, -2.6]
        self.cp = np.array([c1, c1, c2, c3, c1]).T.copy()
        self.pw3p = nppw.PerPwPoly.new(self.xk, self.cp)
        c0 = [1., 0.]
        c1 = [1., 0.]
        c2 = [1., -2.]
        c3 = [-3., 8.]
        c4 = [ 5., 0.]
        self.cl = np.array([c0, c1, c2, c3, c4]).T.copy()
        c4x = [ 5., 8.]
        self.clx = np.array([c0, c1, c2, c3, c4x]).T.copy()
        c3p = [-3., 4.]
        self.clp = np.array([c1, c1, c2, c3p, c1]).T.copy()

    def tearDown(self):
        pass

    def test_spline(self):
        """Check spline."""
        pw = nppw.spline(self.xk, self.yk)
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(self.c, pw.c), "spline failed")
        pw = nppw.spline(self.xk, self.yk, per=1)
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(self.cp, pw.c), "spline per failed")
        pw = nppw.spline(self.xk, self.yk, lo=(None,0), hi=(None,0))
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(self.c, pw.c), "spline hilo failed 1")
        pw = nppw.spline(self.xk[::-1], self.yk[::-1],
                         lo=(None,0), hi=(None,0), extrap=(1,1))
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(self.c, pw.c), "spline rev failed")
        pw = nppw.spline(self.xk, self.yk, lo=1, hi=2)
        self.assertTrue(np.allclose(self.c[0], pw.c[0]) and
                        np.allclose([1.,2.], pw.c[1,1:5:3]),
                        "spline hilo failed 2")
        pw = nppw.spline(self.xk, self.yk, lo=(1,0))
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(self.c, pw.c), "spline lo failed")
        pw = nppw.spline(self.xk, self.yk, hi=(10,0))
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(self.c, pw.c), "spline hi failed")
        pw = nppw.spline(self.xk, self.yk, extrap=1)
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(self.c, pw.c), "spline extrap failed")
        pw = nppw.pline(self.xk, self.yk)
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(self.cl, pw.c), "pline failed")
        pw = nppw.pline(self.xk, self.yk, 1)
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(self.clx, pw.c), "pline extrap failed")
        pw = nppw.pline(self.xk, self.yk, per=1)
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(self.clp, pw.c), "pline per failed")

    def test_splfit(self):
        """Check splfit."""

############################################################################

if __name__ == '__main__':
    unittest.main()
