# Causal reasoning from text

We use the Corr2Cause dataset to test whether a causal claim follows from a set of statements about correlation and independence. Each example has a premise, a hypothesis, and a label. A label of 1 means the hypothesis follows from the premise.

## What we did

1. **Phase 1 · Classical ML.** We turned each example into 28 features and compared Logistic Regression, Linear SVM, and Random Forest. The calibrated graph classifier scored **0.6357 test macro F1** on the corrected run.
2. **Phase 2 · Deep learning.** We fine-tuned DistilBERT on 2,100 training examples and trained a Random Forest on the same examples for a fair comparison. Test macro F1 was **0.5927 for matched ML**, **0.6726 for fine-tuned DistilBERT**, and **0.6718 for the average of their scores**. The average scored highest on development data. DistilBERT alone scored highest on test data.
3. **Graphs and Bayesian queries.** Accepted direct-cause claims become arrows in a separate graph for each premise. We check graph consistency and use explicitly supplied probabilities to demonstrate what happens when we intervene in a small network.

## Start here

| If you want to… | Open |
| --- | --- |
| See the demo with its results already displayed | [Demo notebook](notebooks/project_demo.ipynb) |
| Explain the project in class | [Three-minute demo and viva guide](docs/phase2/VIVA_AND_DEMO.md) |
| Show the Phase 2 submission | [IEEE report](docs/phase2/Report.pdf) · [Slides](docs/phase2/Presentation.pptx) |
| Review the original Phase 1 submission | [Report](docs/phase1_submission/Report.pdf) · [Slides](docs/phase1_submission/AML_Phase1_Project_Presentation.pptx) |
| Check the measured results | [Phase 1](results/phase1/RESULTS.md) · [Phase 2](results/phase2/RESULTS.md) |

## Run the saved demo

From the repository folder, using Python 3.13 on macOS or Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-phase2.txt
python scripts/restore_results.py
python phase2.py demo
python run_tests.py
```

The demo uses the saved measurements in `results/`. You can run it without training or downloading the dataset or model weights. The notebook includes its original experiment outputs and can be read on GitHub. To rerun its cells, restore the results and select the project's Python environment as the notebook kernel.

## Where to find the code

| Path | Purpose |
| --- | --- |
| `masterNotebook.py` | Run Phase 1 from the command line |
| `phase2.py` | Run Phase 2 preparation, training, evaluation, and demo |
| `src/` | Feature extraction, models, graphs, and Bayesian queries |
| `configs/`, `data/`, `corr2cause/` | Experiment settings, demo assumptions, and the official dataset version |
| `scripts/`, `tests/` | Download, demo setup, and checks |
| `docs/`, `notebooks/`, `results/` | Reports, slides, walkthrough, and measured results |

For full training, see the [reproduction commands](docs/phase2/RUN_AND_EXPLAIN.md). Training writes to `outputs/`. The [original Phase 1 GitHub version](https://github.com/Ritik-rao8/AML_DL_Project/tree/7363b7b834ac6c9814eca935ef152651b9d720b0) is available in the repository history.
