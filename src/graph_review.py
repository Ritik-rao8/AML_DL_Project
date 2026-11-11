"""Explicit, label-blind review of small systems under Markov/faithfulness assumptions."""
from functools import lru_cache
from itertools import combinations, product
import re
import networkx as nx
import numpy as np
from src.relations import system_id, parse_claim, resolve_cycles
from src.bayes import validate_queries


def independence_statements(premise):
    statements=set()
    for u,v in re.findall(r'\b([A-F]) is independent of ([A-F])',premise):
        statements.add((tuple(sorted((u,v))),()))
    for u,v,given in re.findall(r'\b([A-F]) and ([A-F]) are independent given ([A-F](?:,? (?:and )?[A-F])*)',premise):
        statements.add((tuple(sorted((u,v))),tuple(sorted(re.findall(r'\b[A-F]\b',given)))))
    return statements


@lru_cache(maxsize=128)
def compatible_graphs(premise):
    variables=sorted(set(re.findall(r'\b[A-F]\b',premise)))
    if len(variables)>6:raise ValueError('Review supports the benchmark systems of at most 6 variables')
    indeps=independence_statements(premise)
    pairs=list(combinations(variables,2))
    # Under the stated assumptions, an adjacent pair has no separating set.
    # Determine the skeleton first, then enumerate directions rather than all graphs.
    separated={pair for pair,given in indeps}
    skeleton=[pair for pair in pairs if pair not in separated]
    graphs=[]
    for bits in product([False,True],repeat=len(skeleton)):
        graph=nx.DiGraph();graph.add_nodes_from(variables)
        graph.add_edges_from((u,v) if bit else (v,u) for (u,v),bit in zip(skeleton,bits))
        if not nx.is_directed_acyclic_graph(graph):continue
        valid=True
        for u,v in pairs:
            others=[x for x in variables if x not in (u,v)]
            for count in range(len(others)+1):
                for given in combinations(others,count):
                    if nx.is_d_separator(graph,{u},{v},set(given)) != ((tuple(sorted((u,v))),tuple(given)) in indeps):
                        valid=False;break
                if not valid:break
            if not valid:break
        if valid:graphs.append(frozenset(graph.edges))
    return tuple(graphs)


def review_edges(premise,candidates):
    graphs=compatible_graphs(premise);result=[]
    for u,v,score in candidates:
        support=sum((u,v) in g for g in graphs)
        status='approved' if graphs and support==len(graphs) else 'rejected' if graphs and support==0 else 'uncertain'
        result.append({'source':u,'target':v,'score':float(score),'status':status,
                       'compatible_DAGs':len(graphs),'DAGs_with_edge':support,
                       'reason':'Edge in every compatible DAG' if status=='approved' else 'Not established by the complete independence constraints'})
    return result


def query_graph(edges, cpds=None):
    """Supplied CPDs and independent enumeration verify graph-to-BN integration."""
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import CausalInference
    if not edges:raise ValueError('No approved edges; no synthetic fallback')
    graph=nx.DiGraph(edges)
    if not nx.is_directed_acyclic_graph(graph):raise ValueError('Approved graph is cyclic')
    if cpds is None:raise ValueError('Explicit CPDs are required; none were supplied')
    if set(cpds)!=set(graph.nodes):raise ValueError('CPD nodes differ from graph')
    bn=DiscreteBayesianNetwork(edges)
    for node in sorted(graph.nodes):
        parents=sorted(graph.predecessors(node));states=list(product([0,1],repeat=len(parents)))
        if cpds[node]['parents']!=parents:raise ValueError('CPD parents differ from graph')
        probabilities=cpds[node]['p_high']
        if len(probabilities)!=len(states) or any(not 0<=p<=1 for p in probabilities):raise ValueError('Invalid CPD values')
        bn.add_cpds(TabularCPD(node,2,[[1-p for p in probabilities],probabilities],
                    evidence=parents or None,evidence_card=[2]*len(parents) or None))
    assert bn.check_model()
    cause,target=sorted(edges)[0]
    actual=float(CausalInference(bn).query([target],do={cause:1},show_progress=False).values[1])
    nodes=sorted(graph.nodes);expected=0.
    for state in product([0,1],repeat=len(nodes)):
        assignment=dict(zip(nodes,state))
        if assignment[cause]!=1 or assignment[target]!=1:continue
        probability=1.
        for node in nodes:
            if node==cause:continue
            c=cpds[node];index=0
            for parent in c['parents']:index=2*index+assignment[parent]
            p=c['p_high'][index];probability*=p if assignment[node] else 1-p
        expected+=probability
    if not np.isclose(actual,expected,atol=1e-10):raise AssertionError('BN differs from independent enumeration')
    return {'cpds':cpds,'cpd_source':'Supplied demonstration assumptions, not fitted to observations',
            'query':f'P({target}=1 | do({cause}=1))','actual':actual,'expected_enumeration':expected,'passed':True}


def review_demo(frame,scores,threshold):
    groups={}
    for row,score in zip(frame.itertuples(),scores):
        if row.num_variables>6:continue
        key=system_id(row.premise)
        group=groups.setdefault(key,{'premise':row.premise,'candidates':{}})
        relation,u,v=parse_claim(row.hypothesis)
        if relation=='direct_cause' and score>=threshold:
            group['candidates'][u,v]=max(float(score),group['candidates'].get((u,v),0.))
    cases=[]
    for key,g in sorted(groups.items()):
        candidates=[(u,v,s) for (u,v),s in sorted(g['candidates'].items())]
        review=review_edges(g['premise'],candidates)
        approved=[(r['source'],r['target']) for r in review if r['status']=='approved']
        cases.append({'systemId':key,'premise':g['premise'],'review':review,'approved_edges':approved,
                      'outcome':'approved edges available' if approved else 'abstain: no approved direct edges'})
    # Small cases chosen by explicit dev-only rule; include approvals and abstentions.
    ordered=sorted(cases,key=lambda c:c['systemId'])
    selected=[]
    for pool in [[c for c in ordered if c['approved_edges']],[c for c in ordered if not c['approved_edges']],ordered]:
        for case in pool:
            if case not in selected:
                selected.append(case);break
    cases=selected[:3]
    integration=None
    for case in cases:
        if case['approved_edges']:
            cpds=supplied_demo_cpds(case['approved_edges'])
            integration=query_graph(case['approved_edges'],cpds)
            integration['assumptions_file']='data/graph_reviews/cpd_assumptions.json'
            break
    return {'reviewer':'Automated exhaustive DAG consistency audit; not a human reviewer',
        'assumptions':['Closed system','Acyclic causal graph','Causal Markov and faithfulness','Premise lists all independences'],
        'cases':cases,'integration':integration,'known_network_queries':validate_queries(),
        'missing_CPD_policy':'Refuse numerical effects without explicit CPDs'}


def supplied_demo_cpds(edges):
    """Materialize the explicitly declared demo assumptions; never fit from labels."""
    import json
    from pathlib import Path
    spec=json.loads((Path(__file__).resolve().parents[1]/'data/graph_reviews/cpd_assumptions.json').read_text())
    graph=nx.DiGraph(edges);cpds={}
    for node in sorted(graph.nodes):
        parents=sorted(graph.predecessors(node))
        low=spec['nonroot_p_high_when_all_parents_low'];high=spec['nonroot_p_high_when_all_parents_high']
        probabilities=[spec['root_p_high'] if not parents else low+(high-low)*sum(s)/len(parents) for s in product([0,1],repeat=len(parents))]
        cpds[node]={'parents':parents,'p_high':probabilities}
    return cpds
