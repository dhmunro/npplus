import unittest
import numpy as np

import npplus as npp
import npplus.solveper as nppsp

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

############################################################################

if __name__ == '__main__':
    unittest.main()
