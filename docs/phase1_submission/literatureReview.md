# Literature Review: Neuro-Symbolic Causal Inference via Language Model Knowledge Extraction and Bayesian Networks

## 1. Introduction

This review surveys the foundational concepts underpinning Phase 1 of the project: extracting causal relations from structured text and encoding them as a Bayesian network for probabilistic causal reasoning. The review is anchored in two key references—Pearl's seminal work on probabilistic reasoning and the BERT architecture that motivates Phase 2—along with the Corr2Cause dataset that serves as our evaluation benchmark.

## 2. Causal Graphs and Directed Acyclic Graphs (DAGs)

A **causal graph** is a directed graph where nodes represent random variables and directed edges represent direct causal influence (Pearl, 1988). When this graph is acyclic—meaning no variable can cause itself through any chain of directed edges—it is called a **directed acyclic graph (DAG)**. DAGs are the structural foundation of causal reasoning because they encode assumptions about which variables influence which others, and critically, which variables are *not* directly related.

Pearl (1988) formalised the connection between DAGs and probability distributions: a DAG *G* over variables *V = {V₁, ..., Vₙ}* implies that the joint distribution *P(V)* factorises as:

> P(V₁, V₂, ..., Vₙ) = ∏ᵢ P(Vᵢ | Pa(Vᵢ))

where *Pa(Vᵢ)* denotes the parents of *Vᵢ* in the graph. This **Markov factorisation** is the fundamental property that makes Bayesian networks computationally tractable: instead of specifying an exponentially large joint distribution, we specify only local conditional probability tables (CPTs) for each node given its parents.

## 3. Bayesian Networks

A **Bayesian network** (BN) is a pair *(G, Θ)* where *G* is a DAG and *Θ* is a set of conditional probability distributions, one for each node (Pearl, 1988). Bayesian networks are powerful because they:

1. **Compactly represent** joint distributions via the Markov factorisation
2. **Support efficient inference** through algorithms like variable elimination and belief propagation
3. **Encode conditional independence** assumptions that can be read directly from the graph structure
4. **Enable causal reasoning** when the edges are given a causal interpretation

### 3.1 Conditional Independence and the Markov Property

A Bayesian network satisfies the **local Markov property**: every variable is conditionally independent of its non-descendants given its parents. This means:

> Vᵢ ⊥ NonDesc(Vᵢ) | Pa(Vᵢ)

This property is what allows the joint distribution to factorise as described above. It also means that by examining the graph structure alone, we can determine many conditional independence relations among the variables without computing any probabilities.

## 4. d-Separation

**d-separation** (directional separation) is the graphical criterion introduced by Pearl (1988) for reading conditional independence relations from a DAG. Two sets of variables *X* and *Y* are **d-separated** by a set *Z* if and only if every path between any node in *X* and any node in *Y* is "blocked" by *Z*. A path is blocked if it contains:

1. **A chain** (A → B → C) or **a fork** (A ← B → C) where the middle node *B* is in *Z*, OR
2. **A collider** (A → B ← C) where neither *B* nor any descendant of *B* is in *Z*

If *X* and *Y* are d-separated given *Z* in the DAG, then *X* and *Y* are conditionally independent given *Z* in every distribution compatible with the DAG:

> X ⊥_d Y | Z in G  ⟹  X ⊥ Y | Z in P

This is the **soundness** of d-separation. The converse (completeness, or **faithfulness**) assumes that the distribution has no "accidental" independencies beyond those implied by the graph—a standard assumption in causal discovery.

d-separation is central to the Corr2Cause dataset: each premise describes a set of statistical relations (correlations and conditional independencies) among variables, and the task is to determine whether a causal hypothesis is consistent with those relations.

## 5. The do-Operator vs. Ordinary Conditioning

A critical distinction in causal inference is between **observational conditioning** (P(Y | X = x)) and **interventional queries** (P(Y | do(X = x))). Pearl's **do-calculus** (Pearl, 1988; Pearl, 2000) formalises this distinction:

- **P(Y | X = x)** answers: "What is the probability of Y among cases where we *observe* X = x?" This reflects a passive observation and does not remove confounding.
- **P(Y | do(X = x))** answers: "What is the probability of Y if we *intervene* to set X = x?" This simulates an experiment by surgically removing all incoming edges to X in the graph (the **graph surgery** or **truncation formula**).

The interventional distribution is computed via the **adjustment formula** (also called the truncated factorisation):

> P(Y | do(X = x)) = ∑_z P(Y | X = x, Pa(X) = z) · P(Pa(X) = z)

when the parents of X are sufficient for adjustment. More generally, Pearl's three rules of do-calculus provide a complete set of transformations for reducing interventional queries to observational quantities when possible.

This distinction is why correlation does not imply causation: P(Y | X = x) can differ from P(Y | do(X = x)) whenever there are confounders (common causes of X and Y). A Bayesian network with a causal interpretation can, in principle, answer both types of queries, making it a more powerful tool than a classifier that only learns correlations.

## 6. Correlation Does Not Imply Causation

The Corr2Cause dataset (Jin et al., 2023) is specifically designed to test whether models can distinguish between correlation and causation. The premise that "A correlates with B" is consistent with multiple causal structures:

- A causes B (A → B)
- B causes A (B → A)
- A common cause C exists (A ← C → B)
- A collider exists with conditioning (A → C ← B, conditioned on C)
- Or combinations of these

Only by examining the *full* set of correlations and conditional independencies can one narrow down the possible causal structures. This is the foundation of **constraint-based causal discovery** algorithms like PC and IC (Pearl, 1988; Spirtes et al., 2000), which use d-separation tests on data to learn the DAG structure.

## 7. Relation Extraction from Text

**Relation extraction** (RE) is the NLP task of identifying semantic relationships between entities mentioned in text. For this project, we extract causal relations (e.g., "A causes B", "A and B are independent given C") from the structured premises in the Corr2Cause dataset.

Traditional approaches to RE include:

- **Rule-based / Pattern-based**: Using dependency parse patterns and lexical cues (e.g., "correlates with", "independent of", "causes") to extract triples. These are high-precision but may miss complex expressions.
- **Feature-based ML classifiers**: Extracting features (dependency paths, token distances, POS tags, entity types) from candidate entity pairs and training a classifier (logistic regression, SVM, random forest) to predict the relation type.
- **Named Entity Recognition (NER)**: Identifying entity mentions (in our case, variable names like A, B, C) as a prerequisite for relation extraction.

These methods form the Phase 1 baseline. Their key limitation is that they extract relations independently and cannot enforce global structural constraints like acyclicity—this is precisely the gap that DAG construction and Bayesian network fitting address.

## 8. The Gap: From Text Extraction to Causal Reasoning

Text extraction can propose plausible causal relations, but it cannot:

1. **Enforce acyclicity**: Independent extraction may produce cycles (A → B → C → A), which are inconsistent with causal DAGs
2. **Quantify uncertainty**: Binary extraction does not provide calibrated probabilities for edge presence
3. **Answer intervention queries**: A list of triples cannot compute P(Y | do(X = x)) without a full probabilistic model
4. **Ensure consistency**: Independently extracted edges may contradict the conditional independence constraints in the data

This gap motivates the pipeline architecture: extraction → DAG construction (with cycle resolution) → Bayesian network fitting → causal queries. The Bayesian network stage provides the formal probabilistic framework that text extraction alone lacks.

## 9. BERT and the Phase 2 Motivation

Devlin et al. (2019) introduced **BERT** (Bidirectional Encoder Representations from Transformers), which learns contextualised word representations by pre-training on large corpora with masked language modelling and next-sentence prediction. BERT and its variants (including DistilBERT) achieve state-of-the-art performance on many NLP tasks by capturing deep semantic relationships in text.

For this project, BERT-family models are relevant as the **Phase 2 (Deep Learning)** approach: fine-tuning a transformer to directly classify whether a causal hypothesis follows from a premise, potentially outperforming the traditional ML features used in Phase 1. However, **BERT is not implemented in Phase 1**; it serves only as the motivation for the DL comparison that will follow.

## 10. Theoretical Rigor Section

### 10.1 d-Separation (Formal Definition)

Let *G = (V, E)* be a DAG. A path *π* between nodes *u* and *v* is **blocked** by a set *Z ⊆ V* if there exists a node *w* on *π* such that either:
- *w* is a non-collider on *π* (chain or fork) and *w ∈ Z*, or
- *w* is a collider on *π* (both edges point into *w*) and *w ∉ Z* and no descendant of *w* is in *Z*.

Two sets *X, Y ⊆ V* are **d-separated** given *Z* in *G*, written *X ⊥_d Y | Z*, if every path between a node in *X* and a node in *Y* is blocked by *Z*.

**Theorem (Soundness and Completeness)**: Under the faithfulness assumption, *X ⊥ Y | Z* in distribution *P* if and only if *X ⊥_d Y | Z* in the DAG *G* compatible with *P*.

### 10.2 The do-Operator (Formal Definition)

Given a causal DAG *G* over variables *V*, the intervention *do(X = x)* produces a new graph *G_x* obtained by removing all edges into *X* (graph surgery). The interventional distribution is:

> P(V \ X | do(X = x)) = ∏_{Vᵢ ∈ V \ X} P(Vᵢ | Pa_G(Vᵢ))   evaluated at X = x

This differs from conditioning because it removes the influence of X's causes, simulating a randomised experiment.

### 10.3 Bayesian Network Joint Factorisation

A Bayesian network *(G, Θ)* represents the joint distribution:

> P(V₁, ..., Vₙ) = ∏ᵢ₌₁ⁿ P(Vᵢ | Pa_G(Vᵢ))

where each factor *P(Vᵢ | Pa_G(Vᵢ))* is specified by a conditional probability table (CPT) *θᵢ ∈ Θ*. The total number of parameters is ∑ᵢ (|Val(Vᵢ)| - 1) · ∏_{Vⱼ ∈ Pa(Vᵢ)} |Val(Vⱼ)|, which is typically much smaller than the (∏ᵢ |Val(Vᵢ)| - 1) parameters needed for the full joint, making BNs an efficient representation.

## References

1. Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann.
2. Pearl, J. (2000). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*.
4. Jin, Z., Liu, J., Lyu, Z., Poff, S., Sachan, M., Mihalcea, R., Diab, M., & Schölkopf, B. (2023). Can Large Language Models Distinguish Correlation from Causation? *arXiv:2306.05836*.
5. Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.
6. Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.

---

*Note: All citations above are to published, verifiable works. References 1–3 are the suggested pre-reads from the Project Ideas PDF. Reference 4 is the Corr2Cause dataset paper. References 5–6 are supplementary standard references in the field.*
