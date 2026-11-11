# Phase 2 results

Train subset: 2100 records in whole premise groups; dev 984; test 1162.
This is a local compute-budget experiment, not a full-205,734-row DL training run.

| Model | Dev macro F1 | Test macro F1 | Positive P | Positive R |
|---|---:|---:|---:|---:|
| Matched ML | 0.5788 | 0.5927 | 0.3431 | 0.2611 |
| Full-train ML (context only) | 0.6534 | 0.6357 | 0.3510 | 0.4778 |
| main | 0.6844 | 0.6726 | 0.3739 | 0.7333 |
| frozen | 0.5917 | 0.5679 | 0.3040 | 0.2111 |
| unweighted | 0.6510 | 0.6443 | 0.3408 | 0.6722 |
| Hybrid matched ML + DL | 0.6871 | 0.6718 | 0.3725 | 0.7389 |

Demo classifier selected on development data: Hybrid matched ML + DL.
Full-train ML is contextual only. Fair budget comparison uses Matched ML and the same DL training rows.
Graph scores measure entailed direct relations, with cycle removal evaluated separately.
Demo review uses an explicit finite-DAG constraint check on small systems; it is automated, not human approval.
CPDs in the numerical demo are supplied assumptions, not estimated from Corr2Cause.
Integration graph source: Hybrid matched ML + DL.
One seed (42); no claim of statistically established superiority or real-world causal discovery.
