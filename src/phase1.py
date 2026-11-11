"""Reproducible Phase 1 only: train/dev selection, held-out evaluation, typed graphs."""
import argparse
import hashlib
import importlib
import json
import pickle
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from src.config import PROJECT_ROOT, SEED
from src.relations import parse_claim, system_id, build_graphs
from src.bayes import validate_queries

ROOT = Path(PROJECT_ROOT)
OUT = ROOT / 'outputs' / 'phase1'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def prepare():
    loader = importlib.import_module('src.data')
    extractor = importlib.import_module('src.features')
    frames, manifests = {}, {}
    for split in ['train', 'dev', 'test']:
        path = ROOT / 'corr2cause' / f'{split}.csv'
        if not path.exists():
            raise FileNotFoundError(f'{path} missing. Run python scripts/download_data.py')
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        cache = OUT / 'cache' / f'{split}.pkl'
        # Include preparation code in cache identity, so fixes invalidate old features.
        code_hash = hashlib.sha256((Path(loader.__file__).read_bytes() + Path(extractor.__file__).read_bytes())).hexdigest()
        key = {'raw_sha256': digest, 'code_sha256': code_hash}
        keyfile = cache.with_suffix('.json')
        if cache.exists() and keyfile.exists() and json.loads(keyfile.read_text()) == key:
            df, features = pd.read_pickle(cache)
        else:
            df = loader.loadAndProcess(str(path), split)
            features = extractor.extractAllFeatures(df, split)
            cache.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle((df, features), cache)
            write_json(keyfile, key)
        frames[split] = (df, features)
        manifests[split] = {'sha256': digest, 'rows': len(df), 'positive': int(df.label.sum())}
    audit = {}
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        da, db = frames[a][0], frames[b][0]
        exact = len(set(da.input) & set(db.input))
        premises = len(set(da.premise) & set(db.premise))
        audit[f'{a}_{b}'] = {'exact_text_overlap': exact, 'shared_premises': premises}
        if exact:
            raise ValueError(f'Exact text leakage between {a} and {b}')
    # Keep official data intact, exclude shared train/dev and dev/test premises
    # from development selection. Test examples never select models or thresholds.
    excluded = set(frames['train'][0].premise) | set(frames['test'][0].premise)
    mask = ~frames['dev'][0].premise.isin(excluded)
    audit['development_selection_rows'] = int(mask.sum())
    write_json(OUT / 'data_manifest.json', {'splits': manifests, 'overlap': audit})
    eda(frames)
    return frames, mask


def eda(frames):
    """Small, regenerated data summary and plots; no duplicate screenshot copies."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, (split, (df, features)) in zip(axes, frames.items()):
        lengths = df.input.str.len()
        summary[split] = {'rows': len(df), 'label_counts': {str(k): int(v) for k,v in df.label.value_counts().items()},
                          'missing': int(df[['input', 'label', 'template', 'num_variables']].isna().sum().sum()),
                          'duplicate_texts': int(df.input.duplicated().sum()),
                          'median_characters': float(lengths.median()),
                          'max_characters': int(lengths.max()),
                          'feature_count': len(features.columns)-3}
        df.label.value_counts().sort_index().plot.bar(ax=ax, title=split)
        ax.set(xlabel='Label', ylabel='Examples')
    fig.tight_layout(); fig.savefig(OUT / 'label_distribution.png'); plt.close(fig)
    write_json(OUT / 'eda.json', summary)


def fingerprint():
    # Model reuse is valid only for the same dataset, features and training code.
    from importlib.metadata import version
    paths = [Path(__file__), ROOT / 'src/config.py', ROOT / 'src/relations.py', ROOT / 'src/bayes.py', ROOT / 'src/rules.py',
             ROOT / 'src/data.py', ROOT / 'src/features.py', ROOT / 'requirements.txt']
    payload = (OUT / 'data_manifest.json').read_bytes() + b''.join(p.read_bytes() for p in paths)
    payload += json.dumps({n: version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'pgmpy']}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def metrics(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)), 'macroF1': float(f1_score(y, pred, average='macro')),
            'per_class': classification_report(y, pred, output_dict=True, zero_division=0)}


def train(frames, mask):
    cols = [c for c in frames['train'][1] if c not in {'label', 'docId', 'template'}]
    x = {s: pair[1][cols] for s, pair in frames.items()}
    y = {s: pair[1].label for s, pair in frames.items()}
    candidates = {
        'LogisticRegression': [make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=SEED)) for c in [.1, 1.]],
        'LinearSVM': [make_pipeline(StandardScaler(), LinearSVC(C=c, class_weight='balanced', dual=False, max_iter=5000, random_state=SEED)) for c in [.1, 1.]],
        'RandomForest': [RandomForestClassifier(n_estimators=100, max_depth=d, class_weight='balanced', n_jobs=2, random_state=SEED) for d in [5, 10]],
    }
    selected = {}
    for name, models in candidates.items():
        best = None
        for model in models:
            start = time.time()
            model.fit(x['train'], y['train'])
            score = float(f1_score(y['dev'][mask], model.predict(x['dev'][mask]), average='macro'))
            print(f'{name} dev macro F1={score:.4f} ({time.time()-start:.1f}s)', flush=True)
            if best is None or score > best['devScore']:
                best = {'model': model, 'devScore': score, 'parameters': str(model)}
        selected[name] = best
    winner = max(selected, key=lambda n: selected[n]['devScore'])
    # Calibrate using training folds only, preserving dev for selection.
    calibrated = CalibratedClassifierCV(selected[winner]['model'], method='sigmoid', cv=3, n_jobs=1)
    calibrated.fit(x['train'], y['train'])
    dev_probabilities = calibrated.predict_proba(x['dev'][mask])[:, 1]
    threshold = max(np.linspace(.05, .95, 19), key=lambda t: (f1_score(y['dev'][mask], dev_probabilities >= t, average='macro'), t))
    modeldir = OUT / 'models'
    modeldir.mkdir(parents=True, exist_ok=True)
    for name, result in selected.items():
        with (modeldir / f'{name}.pkl').open('wb') as handle:
            pickle.dump({'model': result['model'], 'features': cols}, handle)
    with (modeldir / 'graph_model.pkl').open('wb') as handle:
        pickle.dump({'model': calibrated, 'features': cols, 'selected_name': winner}, handle)
    selection = {name: {k: v for k, v in result.items() if k != 'model'} for name, result in selected.items()}
    write_json(OUT / 'selection.json', {'models': selection, 'winner': winner, 'graph_threshold': float(threshold), 'seed': SEED,
                                       'fingerprint': fingerprint(), 'python': platform.python_version(), 'protocol': 'Development macro F1; calibration on training folds; probability threshold selected by development macro F1'})


def evaluate(frames):
    selection = json.loads((OUT / 'selection.json').read_text())
    if selection['fingerprint'] != fingerprint():
        raise ValueError('Data or training code changed; retrain before evaluation')
    test, features = frames['test']
    results = {}
    for path in sorted((OUT / 'models').glob('*.pkl')):
        if path.stem == 'graph_model':
            continue
        with path.open('rb') as f:
            bundle = pickle.load(f)
        results[path.stem] = metrics(test.label, bundle['model'].predict(features[bundle['features']]))
    if len(results) != 3:
        raise FileNotFoundError('Expected three Phase 1 models; run training first')
    # Rule-based implementation remains the historical comparator.
    baseline = importlib.import_module('src.rules')
    results['RuleBased'] = metrics(test.label, test.input.apply(baseline.ruleBasedPredict))
    with (OUT / 'models' / 'graph_model.pkl').open('rb') as f:
        bundle = pickle.load(f)
    probabilities = np.round(bundle['model'].predict_proba(features[bundle['features']])[:, 1], 12)
    threshold = selection['graph_threshold']
    results['CalibratedGraphModel'] = metrics(test.label, (probabilities >= threshold).astype(int))
    claims = []
    for (_, row), probability in zip(test.iterrows(), probabilities):
        if probability < threshold:
            continue
        relation, subject, obj = parse_claim(row.hypothesis)
        claims.append({'systemId': system_id(row.premise), 'subject': subject, 'object': obj,
                       'relation': relation, 'confidence': float(probability), 'docId': row.docId,
                       'sentenceId': row.docId + '_hypothesis', 'hypothesis': row.hypothesis})
    write_json(OUT / 'claims.json', claims)
    graphs = build_graphs(claims)
    graphdir = OUT / 'graphs'
    graphdir.mkdir(parents=True, exist_ok=True)
    for old in graphdir.glob('*.graphml'):
        old.unlink()
    removed = {}
    for key, (graph, dropped) in graphs.items():
        nx.write_graphml(graph, graphdir / f'{key}.graphml')
        removed[key] = dropped
    # This checks mechanics on a known network, not empirical causal effects in Corr2Cause.
    queries = validate_queries()
    write_json(OUT / 'query_validation.json', {'scope': 'Specified synthetic confounded network; not fitted to Corr2Cause', 'queries': queries})
    write_json(OUT / 'cycle_removals.json', removed)
    write_json(OUT / 'metrics.json', results)
    summary = ['# Corrected Phase 1 results', '', '| Model | Accuracy | Macro F1 |', '|---|---:|---:|']
    summary += [f'| {n} | {m["accuracy"]:.4f} | {m["macroF1"]:.4f} |' for n, m in results.items()]
    summary += ['', f'{len(graphs)} separate premise graphs. Only direct claims create edges.',
                'Other accepted claims remain in claims.json without invented direct edges.',
                'Graph edge precision: not measured; Corr2Cause CSVs do not supply complete gold DAGs.',
                'Three numerical queries validated against a specified synthetic network.',
                'No observational CPDs or empirical intervention effects are estimated from Corr2Cause.',
                'These are corrected rerun results. Historical submitted metrics remain in docs/phase1_submission/.']
    (OUT / 'RESULTS.md').write_text('\n'.join(summary) + '\n')
    plot_results(results, graphs)
    print('\n'.join(summary), flush=True)


def plot_results(results, graphs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(list(results), [m['macroF1'] for m in results.values()])
    ax.set(ylabel='Test macro F1', ylim=(0, 1), title='Corrected Phase 1 comparison')
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout(); fig.savefig(OUT / 'model_comparison.png'); plt.close(fig)
    if graphs:
        key = max(sorted(graphs), key=lambda k: graphs[k][0].number_of_edges())
        graph = graphs[key][0]
        fig, ax = plt.subplots(figsize=(7, 5))
        nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=SEED), ax=ax, node_color='#b7dfed', arrows=True)
        ax.set_title('One premise: predicted direct edges (not gold causal structure)')
        ax.axis('off'); fig.tight_layout(); fig.savefig(OUT / 'example_graph.png'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-only', action='store_true', help='Reuse corrected models; do not train')
    parser.add_argument('--prepare-only', action='store_true', help='Audit data and cache features without training')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT / 'SUCCESS.json'
    marker.unlink(missing_ok=True)
    frames, mask = prepare()
    if args.prepare_only:
        return
    if not args.evaluate_only:
        train(frames, mask)
    evaluate(frames)
    write_json(marker, {'completed': True, 'phase': 1, 'timestamp_utc': pd.Timestamp.now(tz='UTC').isoformat()})


if __name__ == '__main__':
    main()
