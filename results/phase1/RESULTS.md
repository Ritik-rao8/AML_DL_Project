# Corrected Phase 1 results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| LinearSVM | 0.6945 | 0.5954 |
| LogisticRegression | 0.6799 | 0.5839 |
| RandomForest | 0.7117 | 0.6055 |
| RuleBased | 0.8184 | 0.4725 |
| CalibratedGraphModel | 0.7823 | 0.6357 |

6 separate premise graphs. Only direct claims create edges.
Other accepted claims remain in claims.json without invented direct edges.
Graph edge precision: not measured; Corr2Cause CSVs do not supply complete gold DAGs.
Three numerical queries validated against a specified synthetic network.
No observational CPDs or empirical intervention effects are estimated from Corr2Cause.
These are corrected rerun results. Historical submitted metrics remain in docs/phase1_submission/.
