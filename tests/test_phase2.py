import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.dl import token_windows,collate,mean_windows
from src.phase2 import graph_benchmark,select_threshold
from src.graph_review import compatible_graphs,review_edges,query_graph,review_demo,supplied_demo_cpds

class TokenizerStub:
    def num_special_tokens_to_add(self,pair):return 3
    def build_inputs_with_special_tokens(self,p,h):return [101]+p+[102]+h+[102]

class Phase2Tests(unittest.TestCase):
    def test_complete_window_coverage(self):
        premise=list(range(10,1040));hypothesis=[77,78]
        windows=token_windows(premise,hypothesis,TokenizerStub())
        self.assertTrue(all(len(w)<=512 for w in windows))
        self.assertEqual([t for w in windows for t in w[1:-4]],premise)
        self.assertTrue(all(w[-3:-1]==hypothesis for w in windows))
        self.assertEqual(len(windows),3)

    def test_hypothesis_overflow_refused(self):
        with self.assertRaises(ValueError):token_windows([1],[2]*510,TokenizerStub())

    def test_pooling_and_padding(self):
        batch=collate([{'windows':[[1,2],[3]],'label':1},{'windows':[[4,5,6]],'label':0}])
        self.assertEqual(batch['attention_mask'].sum().item(),6)
        result=mean_windows(torch.tensor([[1.,2.],[3.,4.],[9.,10.]]),batch['owners'],2)
        self.assertTrue(torch.equal(result,torch.tensor([[2.,3.],[9.,10.]])))

    def test_threshold_uses_only_supplied_labels(self):
        threshold,score=select_threshold([0,0,1,1],[.1,.2,.7,.8])
        self.assertEqual(score,1.)
        self.assertTrue(.2<threshold<=.7)

    def test_graph_includes_empty_references(self):
        frame=pd.DataFrame({'premise':['system one','system one','system two','system two'],
             'hypothesis':['A directly causes B.','B directly causes A.']*2,'num_variables':[2]*4,'label':[1,0,0,0]})
        result=graph_benchmark(frame,np.array([.9,.1,.9,.1]),.5)['summary']
        self.assertEqual(result['systems'],2)
        self.assertEqual(result['before']['tp'],1);self.assertEqual(result['before']['fp'],1)
        self.assertEqual(result['before']['precision'],.5)

    def test_no_predictions_precision_undefined(self):
        frame=pd.DataFrame({'premise':['x','x'],'hypothesis':['A directly causes B.','B directly causes A.'],'num_variables':[2,2],'label':[0,0]})
        result=graph_benchmark(frame,np.array([.1,.1]),.5)['summary']
        self.assertIsNone(result['after']['precision']);self.assertEqual(result['after']['systems_with_edges'],0)

    def test_review_without_reference_labels(self):
        premise='A correlates with B. B correlates with C. A is independent of C.'
        graphs=compatible_graphs(premise)
        self.assertEqual(len(graphs),1)
        result=review_edges(premise,[('A','B',.8),('B','A',.9)])
        self.assertEqual([r['status'] for r in result],['approved','rejected'])

    def test_no_review_approval_for_ambiguous_direction(self):
        self.assertEqual(review_edges('A correlates with B.',[('A','B',.9)])[0]['status'],'uncertain')

    def test_cpd_integration_and_missing_edges(self):
        result=query_graph([('A','B')],supplied_demo_cpds([('A','B')]));self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['actual'],.8)
        with self.assertRaises(ValueError):query_graph([])
        with self.assertRaisesRegex(ValueError,'Explicit CPDs'):query_graph([('A','B')])

    def test_preparation_signature_guard(self):
        import src.phase2 as p
        with tempfile.TemporaryDirectory() as directory,patch.object(p,'OUT',Path(directory)),patch.object(p,'signature',return_value='new'):
            (Path(directory)/'data_manifest.json').write_text(json.dumps({'preparation_signature':'old'}))
            with self.assertRaises(ValueError):p.prepared()

    def test_review_does_not_consult_labels(self):
        premise='A correlates with B. B correlates with C. A is independent of C.'
        frame=pd.DataFrame({'premise':[premise,premise], 'hypothesis':['A directly causes B.','C directly causes B.'], 'num_variables':[3,3], 'label':[0,0]})
        before=review_demo(frame,np.array([.9,.9]),.5)
        frame['label']=[1,1]
        after=review_demo(frame,np.array([.9,.9]),.5)
        self.assertEqual(before,after)
        self.assertTrue(before['integration']['passed'])

    def test_contradictory_reference_labels_refused(self):
        frame=pd.DataFrame({'premise':['x','x'], 'hypothesis':['A directly causes B.']*2, 'num_variables':[2,2], 'label':[0,1]})
        with self.assertRaisesRegex(ValueError,'Contradictory'):
            graph_benchmark(frame,np.array([.8,.8]),.5)

    def test_final_artifacts(self):
        import src.phase2 as p
        if not (p.OUT/'SUCCESS.json').exists():self.skipTest('Run Phase 2 experiments first')
        success=json.loads((p.OUT/'SUCCESS.json').read_text());self.assertEqual(success['signature'],p.signature())
        metrics=json.loads((p.OUT/'metrics.json').read_text());self.assertEqual(len(metrics),6)
        for experiment in ['main','frozen','unweighted']:
            for split in ['dev','test']:
                values=np.load(p.OUT/experiment/(split+'_scores.npy'))
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue(((values>=0)&(values<=1)).all())
        graphs=json.loads((p.OUT/'graph_metrics.json').read_text())
        self.assertTrue(all(g['summary']['systems']==18 for g in graphs.values()))
        demo=json.loads((p.OUT/'demo.json').read_text())
        self.assertTrue(demo['integration']['passed'])
        self.assertTrue(all(q['passed'] for q in demo['known_network_queries']))
        manifest=json.loads((p.OUT/'data_manifest.json').read_text())
        sets=[set(manifest['splits'][s]['docIds']) for s in ['train','dev','test']]
        self.assertTrue(sets[0].isdisjoint(sets[1]));self.assertTrue(sets[0].isdisjoint(sets[2]))
