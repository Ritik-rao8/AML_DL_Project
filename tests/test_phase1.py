import importlib
import json
from pathlib import Path
import tempfile
import unittest
import networkx as nx
from src.relations import parse_claim, system_id, build_graphs, resolve_cycles
from src.bayes import validate_queries

class Phase1Tests(unittest.TestCase):
    def test_direct_direction(self):
        self.assertEqual(parse_claim('B directly causes A.'), ('direct_cause', 'B', 'A'))
        self.assertEqual(parse_claim('A is directly caused by B.'), ('direct_cause', 'B', 'A'))

    def test_no_invented_edges(self):
        for text, kind in [('There exists at least one confounder of A and B.', 'confounder'),
                           ('There exists at least one collider of A and B.', 'collider'),
                           ('A causes something else which causes C.', 'indirect_cause'),
                           ('A is a cause for B, but not a direct one.', 'indirect_cause')]:
            self.assertEqual(parse_claim(text), (kind, None, None))

    def test_separate_systems(self):
        claims = [dict(systemId=k, relation='direct_cause', subject='A', object='B', confidence=.8, docId=k, sentenceId=k) for k in ['one', 'two']]
        graphs = build_graphs(claims)
        self.assertEqual(len(graphs), 2)
        self.assertEqual(set(graphs['one'][0].edges), {('A', 'B')})
        self.assertEqual(graphs['two'][0]['A']['B']['docId'], 'two')

    def test_confidence_and_self_loops(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B', confidence=.9)
        graph.add_edge('B', 'C', confidence=.2)
        graph.add_edge('C', 'A', confidence=.7)
        dag, removed = resolve_cycles(graph)
        self.assertTrue(nx.is_directed_acyclic_graph(dag))
        self.assertEqual((removed[0]['source'], removed[0]['target']), ('B', 'C'))
        self.assertEqual(len(graph.edges), 3)  # does not mutate original
        c = dict(systemId='one', relation='direct_cause', subject='A', object='A', confidence=.9, docId='d', sentenceId='s')
        self.assertEqual(build_graphs([c]), {})

    def test_ties_are_deterministic(self):
        a = nx.DiGraph(); b = nx.DiGraph()
        edges = [('A', 'B'), ('B', 'A')]
        for u, v in edges: a.add_edge(u, v, confidence=.5)
        for u, v in reversed(edges): b.add_edge(u, v, confidence=.5)
        self.assertEqual(set(resolve_cycles(a)[0].edges), set(resolve_cycles(b)[0].edges))

    def test_content_identity(self):
        loader = importlib.import_module('src.data')
        self.assertEqual(loader.generateDocId('same text', 'train', 0), loader.generateDocId('same text', 'test', 50))
        self.assertNotEqual(system_id('premise one'), system_id('premise two'))

    def test_queries_have_known_answers(self):
        results = validate_queries()
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r['passed'] for r in results))
        self.assertNotAlmostEqual(results[0]['actual'], results[1]['actual'])

    def test_missing_data_fails(self):
        from unittest.mock import patch
        import src.phase1 as pipeline
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, 'ROOT', Path(directory)):
            with self.assertRaises(FileNotFoundError): pipeline.prepare()

    def test_stale_models_are_rejected(self):
        from unittest.mock import patch
        import src.phase1 as pipeline
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'selection.json').write_text(json.dumps({'fingerprint': 'old'}))
            with patch.object(pipeline, 'OUT', path), patch.object(pipeline, 'fingerprint', return_value='new'):
                with self.assertRaisesRegex(ValueError, 'retrain'):
                    pipeline.evaluate({})

    def test_saved_outputs(self):
        out = Path(__file__).resolve().parents[1] / 'outputs/phase1'
        if not (out / 'SUCCESS.json').exists():
            self.skipTest('Run the pipeline first for artifact checks')
        manifest = json.loads((out / 'data_manifest.json').read_text())
        self.assertTrue(all(m['rows'] > 0 for m in manifest['splits'].values()))
        self.assertTrue(all(v['exact_text_overlap'] == 0 for k,v in manifest['overlap'].items() if isinstance(v,dict)))
        for path in (out / 'graphs').glob('*.graphml'):
            graph = nx.read_graphml(path)
            self.assertTrue(nx.is_directed_acyclic_graph(graph))
            self.assertFalse(list(nx.selfloop_edges(graph)))
            for _,_,data in graph.edges(data=True):
                self.assertTrue(0 <= data['confidence'] <= 1)
                self.assertTrue(data['docId'])
        self.assertEqual(len(json.loads((out / 'metrics.json').read_text())), 5)
        from src.phase1 import fingerprint
        selection = json.loads((out / 'selection.json').read_text())
        self.assertEqual(selection['fingerprint'], fingerprint())
        self.assertEqual(selection['models'][selection['winner']]['devScore'], max(m['devScore'] for m in selection['models'].values()))
        self.assertTrue(0 < selection['graph_threshold'] < 1)
        self.assertTrue(list((out / 'graphs').glob('*.graphml')))
        claims = json.loads((out / 'claims.json').read_text())
        self.assertTrue(all(c['confidence'] >= selection['graph_threshold'] for c in claims))
        self.assertGreater(len({c['confidence'] for c in claims}), 1)

if __name__ == '__main__': unittest.main()
