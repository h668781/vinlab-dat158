
import numpy as np


def metrics(y, prediction):
    error = np.asarray(prediction) - y
    return {
        'mae': float(np.mean(np.abs(error))),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'r2': float(1 - np.sum(error ** 2) / np.sum((y - y.mean()) ** 2)),
        'within_one': float(np.mean(np.abs(error) <= 1)),
    }


def fit_ridge(X, y, alpha=10):
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1
    Z = np.column_stack([np.ones(len(X)), (X - mean) / scale])
    penalty = np.eye(Z.shape[1]) * alpha
    penalty[0, 0] = 0  # Do not penalise the intercept.
    weights = np.linalg.solve(Z.T @ Z + penalty, Z.T @ y)
    return {'kind': 'ridge', 'mean': mean.tolist(), 'scale': scale.tolist(),
            'weights': weights.tolist(), 'alpha': alpha}


def fit_tree(X, y, max_depth=6, min_leaf=8, max_features=None, seed=42):
    rng = np.random.default_rng(seed)

    def grow(rows, depth):
        values = y[rows]
        leaf = {'value': float(values.mean()), 'n': len(rows)}
        if depth >= max_depth or len(rows) < 2 * min_leaf or np.ptp(values) == 0:
            return leaf
        parent_loss = np.sum((values - values.mean()) ** 2)
        best_loss, best = parent_loss, None
        features = rng.choice(X.shape[1], max_features or X.shape[1], replace=False)
        for feature in features:
            # Exact midpoint candidates, with cumulative sums for split loss.
            order = rows[np.argsort(X[rows, feature], kind='stable')]
            xs, ys = X[order, feature], y[order]
            positions = np.arange(min_leaf, len(rows) - min_leaf + 1)
            positions = positions[xs[positions - 1] < xs[positions]]
            if not len(positions):
                continue
            sums, squares = np.cumsum(ys), np.cumsum(ys ** 2)
            left = squares[positions - 1] - sums[positions - 1] ** 2 / positions
            right_n = len(rows) - positions
            right = squares[-1] - squares[positions - 1] - (
                sums[-1] - sums[positions - 1]) ** 2 / right_n
            loss = left + right
            i = int(np.argmin(loss))
            if loss[i] < best_loss - 1e-10:
                cut = int(positions[i])
                best_loss = float(loss[i])
                best = (int(feature), float((xs[cut - 1] + xs[cut]) / 2),
                        order[:cut], order[cut:])
        if best is None:
            return leaf
        feature, threshold, left_rows, right_rows = best
        return {'feature': feature, 'threshold': threshold,
                'left': grow(left_rows, depth + 1),
                'right': grow(right_rows, depth + 1)}

    return {'kind': 'tree', 'tree': grow(np.arange(len(y)), 0)}


def fit_forest(X, y, n_trees=48, max_depth=8, min_leaf=4, seed=42):
    rng = np.random.default_rng(seed)
    trees = []
    for _ in range(n_trees):
        sample = rng.integers(0, len(y), len(y))
        tree = fit_tree(X[sample], y[sample], max_depth, min_leaf,
                        max_features=4, seed=int(rng.integers(0, 2**31)))
        trees.append(tree['tree'])
    return {'kind': 'forest', 'trees': trees, 'n_trees': n_trees,
            'max_depth': max_depth, 'min_leaf': min_leaf}


def tree_value(node, row):
    while 'value' not in node:
        if row[node['feature']] <= node['threshold']:
            node = node['left']
        else:
            node = node['right']
    return node['value']


def predict(model, X):
    X = np.asarray(X, dtype=float)
    if model['kind'] == 'constant':
        return np.full(len(X), model['value'])
    if model['kind'] == 'ridge':
        Z = (X - np.array(model['mean'])) / np.array(model['scale'])
        return model['weights'][0] + Z @ np.array(model['weights'][1:])
    if model['kind'] == 'tree':
        trees = [model['tree']]
    else:
        trees = model['trees']
    predictions = []
    for row in X:
        tree_predictions = []
        for tree in trees:
            tree_predictions.append(tree_value(tree, row))
        predictions.append(np.mean(tree_predictions))
    return np.array(predictions)
