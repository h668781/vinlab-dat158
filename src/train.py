import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from models import fit_ridge, fit_tree, fit_forest, predict, metrics

ROOT = Path(__file__).resolve().parents[1]
SEED = 158


def save(path, value, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=None if compact else 2,
                              separators=(',', ':') if compact else None,
                              allow_nan=False), encoding='utf-8')


def main():
    data_file = ROOT / 'data/winequality-red.csv'
    with data_file.open() as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        raw = np.array(list(reader), dtype=float)
    if raw.shape != (1599, 12) or not np.isfinite(raw).all():
        raise ValueError('Unexpected dataset schema, row count or missing values')
    # Remove identical feature vectors before splitting, keeping first label.
    # This also prevents duplicate measurements crossing the split boundary.
    _, unique_indices = np.unique(raw[:, :-1], axis=0, return_index=True)
    unique_indices.sort()
    data = raw[unique_indices]
    X, y = data[:, :-1], data[:, -1]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y))
    a, b = int(.6 * len(y)), int(.8 * len(y))
    train, validation, test = order[:a], order[a:b], order[b:]
    save(ROOT / 'results/splits.json', {'seed': SEED, 'index_basis': 'cleaned CSV row order',
         'original_rows_kept': unique_indices.tolist(), 'train': train.tolist(),
         'validation': validation.tolist(), 'test': test.tolist()})

    candidates = [('Median', {'kind': 'constant', 'value': float(np.median(y[train]))})]
    for alpha in [1, 10, 100]:
        model = fit_ridge(X[train], y[train], alpha)
        candidates.append((f'Ridge alpha={alpha}', model))
    for depth in [3, 6, 10]:
        model = fit_tree(X[train], y[train], max_depth=depth)
        candidates.append((f'Tree depth={depth}', model))
    for depth in [5, 8, 12]:
        model = fit_forest(X[train], y[train], max_depth=depth)
        candidates.append((f'Forest depth={depth}', model))
    runs, models = [], []
    for name, model in candidates:
        score = metrics(y[validation], predict(model, X[validation]))
        runs.append({'name': name, 'train': metrics(y[train], predict(model, X[train])),
                     'validation': score})
        models.append(model)
        print(name, 'validation MAE', round(score['mae'], 4), flush=True)
    selected = int(np.argmin([run['validation']['mae'] for run in runs]))
    # Frozen training-only model: no refit on validation or test data.
    model = models[selected]
    test_prediction = predict(model, X[test])
    test_score = metrics(y[test], test_prediction)
    baseline_test = metrics(y[test], predict(models[0], X[test]))
    # Permutation importance uses validation data, not the held-out test data.
    importance = []
    for j, feature in enumerate(header[:-1]):
        scores = []
        for _ in range(5):
            permuted = X[validation].copy()
            permuted[:, j] = rng.permutation(permuted[:, j])
            scores.append(metrics(y[validation], predict(model, permuted))['mae'])
        importance.append({'feature': feature,
                           'mae_increase': float(np.mean(scores) - runs[selected]['validation']['mae'])})
    slices = []
    for quality in np.unique(y[test]):
        mask = y[test] == quality
        slices.append({'quality': int(quality), 'n': int(mask.sum()),
                       'mae': float(np.mean(np.abs(test_prediction[mask] - y[test][mask])))})
    failures = np.argsort(np.abs(test_prediction - y[test]))[-10:][::-1]
    report = {'seed': SEED, 'data_sha256': hashlib.sha256(data_file.read_bytes()).hexdigest(),
              'python': platform.python_version(), 'numpy': np.__version__,
              'raw_rows': len(raw), 'clean_rows': len(data), 'removed': len(raw) - len(data),
              'splits': {'train': len(train), 'validation': len(validation), 'test': len(test)},
              'quality_counts': {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
              'candidates': runs, 'selected': runs[selected]['name'], 'test': test_score,
              'baseline_test': baseline_test, 'quality_slices': slices,
              'permutation_importance_validation': sorted(importance, key=lambda x: -x['mae_increase']),
              'worst_errors': [{'clean_row': int(test[i]), 'actual': float(y[test[i]]),
                               'prediction': float(test_prediction[i])} for i in failures]}
    save(ROOT / 'results/metrics.json', report)
    units = ['g/L', 'g/L', 'g/L', 'g/L', 'g/L', 'mg/L', 'mg/L', 'g/cm³', '', 'g/L', '% vol']
    # These metadata come from training data only.
    features = [{'name': name, 'min': float(X[train, i].min()), 'max': float(X[train, i].max()),
                 'default': float(np.median(X[train, i])), 'unit': units[i]}
                for i, name in enumerate(header[:-1])]
    artifact = {'version': 1, 'selected': report['selected'], 'model': model,
                'features': features, 'test': test_score, 'baseline_test': baseline_test,
                'training_rows': len(train), 'dataset_rows': len(data), 'data_sha256': report['data_sha256']}
    save(ROOT / 'dist/model.json', artifact, compact=True)
    save(ROOT / 'dist/evaluation.json', report)
    fixtures = [{'input': row.tolist(), 'expected': float(value)}
                for row, value in zip(X[test], test_prediction)]
    save(ROOT / 'results/prediction_fixtures.json', fixtures)
    print('Selected:', report['selected'], '\nTest:', test_score)


if __name__ == '__main__':
    main()
