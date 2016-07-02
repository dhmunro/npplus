import unittest

from npplus.lsqfit import regress, levmar, ModelFit

import numpy as np

import __main__

# nosetests --with-coverage --cover-package=npplus

def nearly_eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.shape == b.shape) and np.allclose(a, b)

freq = 1.85
def ftest(p, x, freq=freq):
    p0, p1, p2, p3 = p
    x = np.asfarray(x)
    fxp = freq*x + p2
    return p0 + p1*np.sin(fxp) + p3*np.sin(x)

def gtest(p, x, freq=freq):
    p0, p1, p2, p3 = p
    x = np.asfarray(x)
    fxp = freq*x + p2
    sfxp, sx = np.sin(fxp), np.sin(x)
    yield p0 + p1*sfxp + p3*sx
    yield np.array([np.ones_like(x), sfxp, p1*np.cos(fxp), sx])

class TestLsqFit(unittest.TestCase):
    def setUp(self):
        self.x = np.r_[-2.:5.:100j]
        self.phase, self.freq, self.a = -np.pi/3., freq, 2.1
        self.cs, self.cc = self.a*np.cos(self.phase), self.a*np.sin(self.phase)
        fx = self.freq * self.x
        self.data = self.a * np.sin(fx + self.phase)
        self.pf = [0., self.a, self.phase, 0.]
        self.mdl = [1., np.sin(fx), np.cos(fx), np.sin(self.x)]
        self.pmdl = [0., self.cs, self.cc, 0.]

    def tearDown(self):
        pass

    def test_regress(self):
        """Check regress function."""
        p = regress(self.data, *self.mdl)
        self.assertTrue(nearly_eq(p, self.pmdl), "regress failed 1")
        m = regress(self.data, *self.mdl, errs=0.01, model=1, rcond=1.e-5)
        self.assertTrue(nearly_eq(m.p, self.pmdl), "regress failed 2")
        self.assertTrue(nearly_eq(m(), self.data), "regress model failed")

    def test_levmar(self):
        """Check levmar function."""
        m = levmar(self.data, gtest, [.1,1.,-1.,-.1], self.x, freq=freq)
        self.assertTrue(nearly_eq(m.p, self.pf), "levmar failed g")
        self.assertTrue(nearly_eq(m(self.x,freq), self.data),
                        "levmar model failed g")
        m = levmar(self.data, ftest, [.1,1.,-1.,-.1], self.x, freq)
        self.assertTrue(nearly_eq(m.p, self.pf), "levmar failed f")
        self.assertTrue(nearly_eq(m(self.x,freq=freq), self.data),
                        "levmar model failed f")
        m = levmar(self.data, gtest, [.1,1.,-1.,0.], self.x, freq=freq, pfix=3)
        self.assertTrue(nearly_eq(m.p, self.pf), "levmar failed g pfix")
        self.assertTrue(nearly_eq(m(self.x,freq), self.data),
                        "levmar model failed g pfix")
        m = levmar(self.data, ftest, [.1,1.,-1.,0.], self.x, freq, pfix=3)
        self.assertTrue(nearly_eq(m.p, self.pf), "levmar failed f pfix")
        self.assertTrue(nearly_eq(m(self.x,freq=freq), self.data),
                        "levmar model failed f pfix")
        m = levmar(self.data, gtest, [.1,1.,-1.,0.], self.x,
                   pmin=[-.1,.9,-2.,-1.], pmax=[.3,3.,0.,1.])
        self.assertTrue(nearly_eq(m.p, self.pf), "levmar failed g pmin/max")
        self.assertTrue(nearly_eq(m(self.x,freq), self.data),
                        "levmar model failed g pmin/max")
        m = levmar(self.data, ftest, [.1,1.,-1.,0.], self.x, pfix=3,
                   pmin=[-.1,.9,-2.,-1.], pmax=[.3,3.,0.,1.])
        self.assertTrue(nearly_eq(m.p, self.pf), "levmar failed f pmin/max")
        self.assertTrue(nearly_eq(m(self.x,freq), self.data),
                        "levmar model failed f pmin/max")
        # bad pmin here causes L-M algorithm to increase lambda
        # none of the other tests cause this code to execute
        m = levmar(self.data, gtest, [.1,1.,-1.,0.], self.x,
                   pmin=[.1,.9,-2.,-1.], pmax=[.3,3.,0.,1.])
        self.assertTrue(m.p[0]==0.1, "levmar failed increase lambda check")

    def test_modelfit(self):
        """Check ModelFit class."""
        m = regress(self.data, *self.mdl, model=1)
        perr = [ 0.10122379,  0.14352292,  0.13970133,  0.13635115]
        self.assertTrue(nearly_eq(m.perr, perr), "ModelFit.perr failed")
        cpcov = m.chi2pcov
        z = np.zeros_like(cpcov)
        self.assertTrue(nearly_eq(cpcov, z), "ModelFit.chi2pcov failed")
        cperr = m.chiperr
        z = np.zeros_like(cperr)
        self.assertTrue(nearly_eq(cperr, z), "ModelFit.chi2perr failed")
        cprob = m.chi2prob()
        self.assertTrue(nearly_eq(cprob, 1.), "ModelFit.chi2prob failed")

############################################################################

if __name__ == '__main__':
    unittest.main()
