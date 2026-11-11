"""Known-network validation, separate from unvalidated text-derived graphs."""
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, CausalInference


def validate_queries():
    # Z confounds X -> Y. All probabilities below are specified, not learned.
    bn = DiscreteBayesianNetwork([('Z', 'X'), ('Z', 'Y'), ('X', 'Y')])
    bn.add_cpds(
        TabularCPD('Z', 2, [[.5], [.5]]),
        TabularCPD('X', 2, [[.9, .1], [.1, .9]], evidence=['Z'], evidence_card=[2]),
        TabularCPD('Y', 2, [[.9, .2, .6, .1], [.1, .8, .4, .9]],
                   evidence=['Z', 'X'], evidence_card=[2, 2]))
    assert bn.check_model()
    obs = VariableElimination(bn)
    causal = CausalInference(bn)
    cases = [('P(Y=1 | X=1)', .89, obs.query(['Y'], evidence={'X': 1}, show_progress=False).values[1]),
             ('P(Y=1 | do(X=1))', .85, causal.query(['Y'], do={'X': 1}, show_progress=False).values[1]),
             ('P(Y=1 | do(X=0))', .25, causal.query(['Y'], do={'X': 0}, show_progress=False).values[1])]
    results = []
    for query, expected, actual in cases:
        if not np.isclose(actual, expected, atol=1e-10):
            raise AssertionError(f'{query}: expected {expected}, got {actual}')
        results.append({'query': query, 'expected': expected, 'actual': float(actual), 'passed': True})
    return results
