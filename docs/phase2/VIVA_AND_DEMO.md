# Three-minute demo and viva guide

Open the [demo notebook](../../notebooks/project_demo.ipynb). The [slides](Presentation.pptx) follow the same sequence.

1. **Show one input.** A premise describes correlations and independences in a small system. The hypothesis makes a causal claim such as “A directly causes B.” The model predicts whether the claim follows from that premise.
2. **Compare the models.** Phase 1 used 28 handcrafted features. In Phase 2, matched ML and DistilBERT trained on the same 2,100 examples. Point to test macro F1 in the notebook. Matched ML scored 0.5927, fine-tuned DistilBERT scored 0.6726, and the average scored 0.6718. We chose the average for the demo using development data.
3. **Show one graph and query.** Each premise gets its own graph. The automated reviewer checks which proposed arrows agree with the stated assumptions. In the example, setting A to 1 gives a 0.60 probability that D is 1 under our supplied probability tables. An independent calculation gives the same answer.
4. **Give the scope.** We used synthetic text, one random seed, and a training subset. The graph benchmark checks which direct relations follow from the premises. The Bayesian demo uses supplied probabilities.

## Questions to practice

- **Why macro F1?** It computes F1 separately for each class and averages them equally. This makes performance on the smaller positive class count just as much.
- **Why split text into windows?** DistilBERT accepts up to 512 tokens at a time. Each window pairs a portion of the premise with the entire hypothesis. We average the window representations and make one prediction for the example.
- **What did the ablations change?** The frozen version trains the classifier head while keeping the encoder fixed. The unweighted version fine-tunes the encoder with ordinary loss. The main run gives the rarer class more weight.
- **What makes the comparison fair?** Matched ML and DistilBERT see the same training examples. Development data is used to select settings. The official test split measures the selected settings.
- **How do you handle a cycle?** In the test graphs, remove the lowest-score edge in the cycle and repeat until the graph is acyclic. In the demo, the logical reviewer approves a proposed arrow if it appears in every graph compatible with the premise and the stated assumptions.
- **What does an intervention mean?** Observing A tells us A's value. Setting A directly changes how A gets its value in the assumed network. The two questions can give different answers.

Every team member should be able to walk through an example from the premise to the model prediction, the graph, and the intervention.
