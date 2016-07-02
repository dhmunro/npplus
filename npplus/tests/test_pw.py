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
        self.dydxk = self.c[1,1:]  # for testing PwPoly
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

    def test_pwpoly(self):
        """Check PwPoly class."""
        # note that pline tests PwPoly(x,y)
        pw = nppw.PwPoly(self.xk, self.yk, self.dydxk)
        ts, tr = str(pw), repr(pw)
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(self.c, pw.c), "PwPoly failed")
        y = pw(self.xk)
        yy, dydx = pw(self.xk, 1)
        self.assertTrue(np.allclose(y,yy) and np.allclose(y, self.c[0,1:]) and
                        np.allclose(dydx, self.c[1,1:]), "PwPoly () failed")
        pw = nppw.PwPoly(self.xk[::-1], self.yk[::-1], self.dydxk[::-1])
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(self.c, pw.c), "PwPoly rev failed")
        pw2 = nppw.PwPoly(self.xk, [self.yk, -self.yk],
                          [self.dydxk, -self.dydxk])
        c2 = np.transpose([self.c, -self.c], (1,0,2))
        self.assertTrue(np.allclose(self.xk, pw2.xk) and len(pw2)==2 and
                        np.allclose(c2, pw2.c), "PwPoly multi failed")
        self.assertTrue(len(pw2)==2 and pw2.shape==(2,) and pw2.ndim==1 and
                        pw2.deg==3 and pw.ndim==0 and pw.shape==() and
                        pw.deg==3 and pw.degree()==3,
                        "PwPoly properties failed")
        j = pw.jumps()[-1]
        self.assertTrue(np.allclose(pw.jumps(2), np.zeros((3,4))) and
                        np.allclose(j, [-1., 2.5, -3.5, 2.]),
                        "PwPoly jumps failed")
        pwx = (+pw + 2.25*pw + (0 + pw*0.25)) - pw/2
        self.assertTrue(np.allclose(self.xk, pwx.xk) and
                        np.allclose(3*self.c, pwx.c), "PwPoly +-*s failed")
        pwx = pw2[0]
        self.assertTrue(np.allclose(self.xk, pwx.xk) and
                        np.allclose(self.c, pwx.c), "PwPoly [] failed")
        pwx = -pw2[::-1]
        self.assertTrue(np.allclose(self.xk, pwx.xk) and
                        np.allclose(c2, pwx.c), "PwPoly [] or neg failed")
        pwx = pw2 + pw
        c2 = np.transpose([2*self.c, np.zeros_like(self.c)], (1,0,2))
        self.assertTrue(np.allclose(self.xk, pwx.xk) and
                        np.allclose(c2, pwx.c), "PwPoly multi + failed")
        pwx = pw2 - pw
        c2 = np.transpose([np.zeros_like(self.c), -2*self.c], (1,0,2))
        self.assertTrue(np.allclose(self.xk, pwx.xk) and
                        np.allclose(c2, pwx.c), "PwPoly multi - failed")
        pwx = nppw.pline(self.xk, self.yk).deriv()
        pwy = nppw.PwPoly(self.xk, [0, -2, 8])  # test histogram, two ways
        pwz = nppw.PwPoly(self.xk, [0, 0, -2, 8, 0])
        self.assertTrue(np.allclose(self.xk, pwz.xk) and
                        np.allclose(pwx.c, pwy.c) and np.allclose(pwx.c, pwz.c),
                        "PwPoly histogram or deriv failed")
        pw0 = pw.deriv(4)
        self.assertTrue(np.allclose(self.xk, pw0.xk) and
                        np.allclose(pw0.c, np.zeros((1,5))),
                        "PwPoly deriv(4) failed")
        z, zx = pw.roots(), pwx.roots()
        self.assertTrue(np.allclose(z, [-2., 0.34910147, 2.4702701]) and
                        np.allclose(zx, [-1., 0., 2., 3.]),
                        "PwPoly roots failed")
        ipw2 = pw2.integ(lbnd=0, k=[0.,0.])
        pw2d = ipw2.deriv()
        self.assertTrue(np.allclose(pw2d.c, pw2.c),
                        "PwPoly integ or deriv failed")
        x = np.r_[-1.:3.4:0.5]
        pwp = nppw.PerPwPoly(self.xk, self.yk)
        pwx, pwpx = pw.reknot(x), pwp.reknot(x[::-1])
        y, yx, yp, ypx = pw(x,3), pwx(x,3), pwp(x,3), pwpx(x,3)
        pwy = pw.reknot(x, 2)
        yxx = pwy(x)
        self.assertTrue(np.allclose(y,yx) and np.allclose(yp,ypx) and
                        np.allclose(y[0],yxx), "PwPoly reknot failed")
        qw = pw + pwx
        qwx = 2*pwx
        self.assertTrue(np.allclose(qw.xk, qwx.xk) and
                        np.allclose(qw.c, qwx.c), "PwPoly mixed + failed")
        qw = nppw.pline(self.xk, self.xk)
        qwx, qpw = qw*qw, qw*pw
        y, yx, yp, ypx = qw(x)**2, qwx(x), qw(x)*pw(x), qpw(x)
        self.assertTrue(np.allclose(y,yx) and np.allclose(yp,ypx),
                        "PwPoly*PwPoly failed")

    def test_spline(self):
        """Check spline and pline."""
        pw = nppw.spline(self.xk, self.yk)
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(self.c, pw.c), "spline failed")
        pw = nppw.spline(self.xk, self.yk, per=1)
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(self.cp, pw.c), "spline per failed")
        y, yk = pw(self.xk), self.yk.copy()
        yk[-1] = yk[0]
        self.assertTrue(np.allclose(y, yk), "spline per () failed")
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
        pw = nppw.spline(self.xk, [self.yk, -self.yk])
        c2 = np.transpose([self.c, -self.c], (1,0,2))
        self.assertTrue(np.allclose(self.xk, pw.xk) and
                        np.allclose(c2, pw.c), "spline multi failed")

    def test_plfit(self):
        """Check plfit."""
        pw = nppw.pline(self.xk, self.yk)
        x = np.r_[-1.:3.:100j]
        p = nppw.plfit(self.xk, x, pw(x))
        self.assertTrue(np.allclose(pw.xk, p.xk) and
                        np.allclose(pw.c, p.c), "plfit failed")
        p = nppw.plfit(self.xk, x, pw(x), errs=0.01*(np.absolute(pw(x))+10.))
        self.assertTrue(np.allclose(pw.xk, p.xk) and
                        np.allclose(pw.c, p.c), "plfit errs failed")
        p = nppw.plfit(self.xk, x, pw(x), lo=1.,hi=5.)
        self.assertTrue(np.allclose(pw.xk, p.xk) and
                        np.allclose(pw.c, p.c), "plfit hilo failed")
        pwp = nppw.pline(self.xk, self.yk, per=1)
        pp = nppw.plfit(self.xk, x, pwp(x), per=1)
        self.assertTrue(np.allclose(pwp.xk, pp.xk) and
                        np.allclose(pwp.c, pp.c), "plfit per failed")
        pw = nppw.pline(self.xk, self.yk, extrap=1)
        p = nppw.plfit(self.xk, x, pw(x), extrap=1)
        self.assertTrue(np.allclose(pw.xk, p.xk) and
                        np.allclose(pw.c, p.c), "plfit extrap failed")
        pw = nppw.pline(self.xk, self.yk)
        x = np.r_[-2.:3.5:100j]
        p = nppw.plfit(self.xk, x, pw(x))
        xk = np.concatenate(([-2.], pw.xk, [3.5]))
        c = np.concatenate((pw.c[:,:1], pw.c, pw.c[:,-1:]), axis=1)
        self.assertTrue(np.allclose(xk, p.xk) and
                        np.allclose(c, p.c), "plfit beyond failed")
        x = np.r_[-1.:3.:100j]
        p = nppw.plfit(self.xk, x, [pw(x),-pw(x)])
        c = np.concatenate((pw.c[:,None], -pw.c[:,None]), axis=1)
        self.assertTrue(np.allclose(pw.xk, p.xk) and
                        np.allclose(c, p.c), "plfit multi failed")

    def test_splfit(self):
        """Check splfit."""
        pw = nppw.spline(self.xk, self.yk)
        x = np.r_[-1.:3.:100j]
        p = nppw.splfit(self.xk, x, pw(x))
        self.assertTrue(np.allclose(pw.xk, p.xk) and
                        np.allclose(pw.c, p.c), "splfit failed")
        p = nppw.splfit(self.xk[::-1], x, pw(x))
        self.assertTrue(np.allclose(pw.xk, p.xk) and
                        np.allclose(pw.c, p.c), "splfit reversed failed")
        p = nppw.splfit(self.xk, x, pw(x), nc=1)
        self.assertTrue(np.allclose(pw.xk, p.xk) and
                        np.allclose(pw.c, p.c), "splfit nc=1 failed")
        p = nppw.splfit(self.xk, x, pw(x), nc=0, extrap=1)
        self.assertTrue(np.allclose(pw.xk, p.xk) and
                        np.allclose(pw.c, p.c), "splfit nc=0 failed")
        p, cs = nppw.splfit(self.xk, x, pw(x), cost=1)
        self.assertTrue(np.allclose(np.zeros_like(cs), cs),
                        "splfit cost failed")
        p = nppw.splfit(self.xk, x, pw(x), lo=(1,1),hi=(5,10))
        self.assertTrue(np.allclose(pw.xk, p.xk) and
                        np.allclose(pw.c, p.c), "splfit hilo failed")
        p = nppw.splfit(self.xk, x, pw(x), lo=(2,0),hi=(4,1))
        self.assertTrue(np.allclose(p.c[0:2,1],[2,0]) and
                        np.allclose(p.c[0:2,-1], [4,1]),
                        "splfit hilo failed 2")
        pw = nppw.spline(self.xk, self.yk, per=1)
        p, cs = nppw.splfit(self.xk, x, pw(x), cost=1, per=1)
        self.assertTrue(np.allclose(pw.xk, p.xk) and
                        np.allclose(pw.c, p.c), "splfit per failed")
        p = nppw.splfit(self.xk, x, [pw(x),-pw(x)], per=1)
        c = np.concatenate((pw.c[:,None], -pw.c[:,None]), axis=1)
        self.assertTrue(np.allclose(pw.xk, p.xk) and
                        np.allclose(c, p.c), "splfit multi failed")

############################################################################

if __name__ == '__main__':
    unittest.main()
