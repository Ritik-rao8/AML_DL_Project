# Literature and design rationale

| Source | What it contributes | How used here |
|---|---|---|
| [BERT, Devlin et al.](https://arxiv.org/abs/1810.04805) | Bidirectional pretrained representations with task-specific fine-tuning | Premise–hypothesis representation and a small classifier head |
| [DistilBERT, Sanh et al.](https://arxiv.org/abs/1910.01108) | A smaller distilled BERT encoder | One pretrained checkpoint, shared across windows |
| [Corr2Cause, Jin et al.](https://arxiv.org/abs/2306.05836) | Causal entailment from correlation/independence statements; generalization limits | Fixed benchmark and explicit synthetic-language limitations |
| Pearl, *Causality*, 2000 | DAGs, d-separation and interventions | Relation semantics, review assumptions and do-versus-conditioning explanation |
| [pgmpy examples](https://pgmpy.org/examples/Causal_Inference.html) | Bayesian graph inference and interventional queries | Specified-CPD demonstrations and numerical validation |

The window average and the 50/50 score combination are project design choices.
They are evaluated as simple mechanisms, not claimed as novel algorithms from the
papers. Corr2Cause label 0 means not entailed; it does not necessarily establish
that an edge is impossible in every compatible graph. Entailed-edge precision is
therefore narrower than recovery precision for a fully known generating DAG.

Our weighted/ordinary-loss comparison changes one loss setting. The frozen-encoder
comparison also uses conventional head-only optimization with a different learning
rate/batch size; that optimization difference limits a strictly causal interpretation
of the ablation itself. One seed does not establish statistical superiority.
