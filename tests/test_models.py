"""The math checked on made-up numbers: closed forms against a numeric solver, bonds against known prices."""
import datetime as dt, os, sys, unittest
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(HERE, "portfolio"), os.path.join(HERE, "bonds")]
import constrained, mv_engine as mv, xlbond  # noqa: E402

NAMES = ["A", "B", "C", "D"]
MU = [0.010, 0.008, 0.012, 0.006]
SD = [0.06, 0.045, 0.08, 0.03]
CORR = np.array([[1, .3, .4, .1], [.3, 1, .2, .15], [.4, .2, 1, .05], [.1, .15, .05, 1]])


def universe():
    return mv.universe_from_moments(NAMES, mu=MU, sd=SD, corr=CORR, rf=0.002)


class Portfolios(unittest.TestCase):
    def test_tangency_matches_the_solver(self):
        U = universe()
        closed, (numeric, _) = mv.tangency(U), mv.solve_numeric(U, "max_sharpe")
        np.testing.assert_allclose(closed.weights, numeric.weights, atol=1e-4)

    def test_minimum_variance_weights_sum_to_one(self):
        g = mv.gmv(universe())
        self.assertAlmostEqual(float(np.sum(g.weights)), 1.0, places=10)

    def test_long_only_has_no_shorts(self):
        p = constrained.tangency(universe(), constrained.Constraints(long_only=True))
        w = np.asarray(p.weights)
        self.assertTrue((w >= -1e-9).all())
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)


class Bonds(unittest.TestCase):
    def test_par_bond_prices_at_par(self):
        p = xlbond.price(dt.date(2026, 1, 15), dt.date(2036, 1, 15), 0.05, 0.05)
        self.assertAlmostEqual(p, 100.0, places=8)

    def test_yield_round_trip(self):
        s, m = dt.date(2026, 3, 1), dt.date(2041, 6, 15)
        p = xlbond.price(s, m, 0.04, 0.0537)
        self.assertAlmostEqual(xlbond.yld(s, m, 0.04, p), 0.0537, places=9)

    def test_modified_duration_below_macaulay(self):
        s, m = dt.date(2026, 3, 1), dt.date(2036, 3, 1)
        self.assertLess(xlbond.mduration(s, m, 0.05, 0.06), xlbond.duration(s, m, 0.05, 0.06))


if __name__ == "__main__":
    unittest.main()
