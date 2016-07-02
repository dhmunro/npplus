import unittest

from npplus.pcwise import pcwise

import numpy as np
from numpy.polynomial import Polynomial

import __main__

# nosetests --with-coverage --cover-package=npplus

@pcwise
def bessj0(x):
    """Bessel function J0(x)."""
    pnum = Polynomial([57568490574.0, -13362590354.0, 651619640.7,
                       -11214424.18, 77392.33017, -184.9052456])
    pden = Polynomial([57568490411.0, 1029532985.0, 9494680.718,
                       59272.64853, 267.8532712, 1.0])
    def fsmall(x):
        y = x*x
        return pnum(y) / pden(y)

    pcos = Polynomial([1.0, -0.1098628627e-2, 0.2734510407e-4,
                       -0.2073370639e-5, 0.2093887211e-6])
    psin = Polynomial([ -0.1562499995e-1, 0.1430488765e-3, -0.6911147651e-5,
                         0.7621095161e-6, -0.934935152e-7])
    def fbig(x):
        ax = abs(x)
        z = 8.0/ax
        y = z*z
        x = ax - 0.785398164 # pi/4, rounded incorrectly
        return np.sqrt(0.636619772/ax) * (np.cos(x)*pcos(y)
                                          - np.sin(x)*z*psin(y))

    return (fbig, -8.0, fsmall, 8.0, fbig)

@pcwise
def posj0(x):
    """Bessel function J0(x) for x>0 only."""
    pnum = Polynomial([57568490574.0, -13362590354.0, 651619640.7,
                       -11214424.18, 77392.33017, -184.9052456])
    pden = Polynomial([57568490411.0, 1029532985.0, 9494680.718,
                       59272.64853, 267.8532712, 1.0])
    def fsmall(x):
        y = x*x
        return pnum(y) / pden(y)

    pcos = Polynomial([1.0, -0.1098628627e-2, 0.2734510407e-4,
                       -0.2073370639e-5, 0.2093887211e-6])
    psin = Polynomial([ -0.1562499995e-1, 0.1430488765e-3, -0.6911147651e-5,
                         0.7621095161e-6, -0.934935152e-7])
    def fbig(x):
        ax = abs(x)
        z = 8.0/ax
        y = z*z
        x = ax - 0.785398164 # pi/4, rounded incorrectly
        return np.sqrt(0.636619772/ax) * (np.cos(x)*pcos(y)
                                          - np.sin(x)*z*psin(y))

    return ("blow up for x<=0", 0.0, fsmall, 8.0, fbig, 10., "blow up for x>10")

class TestPcwise(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(-10, 18, 4)
        self.y = np.array([-0.24593576, 0.15064526, 0.22389078, 0.22389078,
                           0.15064526, -0.24593576, 0.17107348])
        self.xx = np.arange(5, 11, 2)
        self.yy = np.array([-0.17759677,  0.30007927, -0.09033361])

    def tearDown(self):
        pass

    def test_pcwise(self):
        """Check pcwise decorator."""
        y = bessj0(self.x)
        self.assertTrue(np.allclose(y, self.y), "bessj0 failed")
        y = bessj0(self.x[3])
        self.assertTrue(np.allclose(y, self.y[3]), "bessj0 scalar failed")
        yy = posj0(self.xx)
        self.assertTrue(np.allclose(yy, self.yy), "posj0 failed")
        with self.assertRaises(ValueError) as cm:
            y = posj0([-1,1,3])
        self.assertEqual(cm.exception.args[0], "blow up for x<=0")
        with self.assertRaises(ValueError) as cm:
            y = posj0([7,9,11])
        self.assertEqual(cm.exception.args[0], "blow up for x>10")

############################################################################

if __name__ == '__main__':
    unittest.main()
