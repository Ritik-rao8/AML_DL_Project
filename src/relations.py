"""Typed claims and graphs; anonymous variables belong to one premise only."""
import hashlib
import re
import networkx as nx


def system_id(premise):
    return hashlib.sha256(premise.strip().encode()).hexdigest()


def parse_claim(hypothesis):
    match = re.fullmatch(r'\s*([A-F]) directly causes ([A-F])\.?\s*', hypothesis)
    if match:
        return 'direct_cause', match.group(1), match.group(2)
    match = re.fullmatch(r'\s*([A-F]) (?:is directly caused by|directly caused by) ([A-F])\.?\s*', hypothesis)
    if match:
        return 'direct_cause', match.group(2), match.group(1)
    lower = hypothesis.lower()
    kind = ('collider' if 'collider' in lower else 'confounder' if 'confounder' in lower
            else 'indirect_cause' if any(s in lower for s in ['not a direct one', 'causes something else', 'ancestor', 'descendant']) else 'unsupported')
    return kind, None, None


def resolve_cycles(graph):
    original = graph
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(original.nodes(data=True)))
    graph.add_edges_from(sorted(original.edges(data=True)))
    removed = []
    while not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        edge = min(cycle, key=lambda e: (graph.edges[e]['confidence'], e[0], e[1]))
        removed.append({'source': edge[0], 'target': edge[1], **dict(graph.edges[edge])})
        graph.remove_edge(*edge)
    return graph, removed


def build_graphs(claims):
    graphs = {}
    for claim in claims:
        if claim['relation'] != 'direct_cause' or claim['subject'] == claim['object']:
            continue
        graph = graphs.setdefault(claim['systemId'], nx.DiGraph())
        u, v = claim['subject'], claim['object']
        if not graph.has_edge(u, v) or claim['confidence'] > graph[u][v]['confidence']:
            graph.add_edge(u, v, confidence=float(claim['confidence']),
                           docId=claim['docId'], sentenceId=claim['sentenceId'])
    return {key: resolve_cycles(g) for key, g in sorted(graphs.items())}
