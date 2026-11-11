"""Re-extract reviewed triples, rebuild their DAG and recompute Bayesian answers."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import networkx as nx
from src.phase2 import OUT,signature
from src.relations import parse_claim,build_graphs
from src.graph_review import query_graph
from src.bayes import validate_queries

def main():
    if json.loads((OUT/'SUCCESS.json').read_text())['signature']!=signature():raise ValueError('Stale saved results')
    recorded=nx.read_graphml(OUT/'reviewed_graph.graphml');triples=[]
    for _,_,data in recorded.edges(data=True):
        relation,subject,obj=parse_claim(data['hypothesis'])
        triples.append({'subject':subject,'object':obj,'relation':relation,'confidence':float(data['confidence']),
                        'systemId':recorded.graph['systemId'],'docId':data['docId'],'sentenceId':data['sentenceId']})
    rebuilt=build_graphs(triples)[recorded.graph['systemId']][0]
    if set(rebuilt.edges)!=set(recorded.edges):raise AssertionError('Re-extracted graph differs')
    demo=json.loads((OUT/'demo.json').read_text())
    result=query_graph(list(rebuilt.edges),demo['integration']['cpds'])
    print(json.dumps({'classifier_scores':'Reused saved predictions; no new neural inference',
                      'triples':triples,'integration':result,'known_queries':validate_queries()},indent=2))

if __name__=='__main__':main()
