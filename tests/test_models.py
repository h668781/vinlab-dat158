import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from models import fit_tree, fit_forest, fit_ridge, predict, metrics


class ModelTests(unittest.TestCase):
    def test_tree_learns_step_and_threshold_boundary(self):
        X = np.arange(20, dtype=float).reshape(-1, 1)
        y = np.where(X[:, 0] < 10, 2., 8.)
        model = fit_tree(X, y, max_depth=2, min_leaf=2)
        np.testing.assert_allclose(predict(model, X), y)
        np.testing.assert_allclose(predict(model, [[9.5], [9.50001]]), [2, 8])

    def test_constant_features_terminate(self):
        model = fit_tree(np.ones((30, 3)), np.arange(30.), max_depth=20)
        self.assertEqual(model['tree']['value'], 14.5)

    def test_ridge_fits_line_and_keeps_constant_feature_finite(self):
        X = np.column_stack([np.arange(20.), np.ones(20)])
        y = 3 * X[:, 0] + 2
        model = fit_ridge(X, y, alpha=0.0001)
        np.testing.assert_allclose(predict(model, X), y, atol=0.001)

    def test_forest_reproducible_and_respects_target_range(self):
        X = np.arange(90.).reshape(30, 3)
        y = np.sin(X[:, 0]) + 5
        first = fit_forest(np.column_stack([X, X]), y, n_trees=5)
        second = fit_forest(np.column_stack([X, X]), y, n_trees=5)
        self.assertEqual(first, second)
        p = predict(first, np.column_stack([X, X]))
        self.assertTrue(np.all((p >= y.min()) & (p <= y.max())))

    def test_metric_definitions(self):
        scores = metrics(np.array([1., 3.]), np.array([2., 3.]))
        self.assertAlmostEqual(scores['mae'], .5)
        self.assertAlmostEqual(scores['rmse'], np.sqrt(.5))
        self.assertAlmostEqual(scores['r2'], .5)


if __name__ == '__main__':
    unittest.main()
