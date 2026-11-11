# Feature Dictionary

This document describes every feature used in the relation extraction classifier, along with the rationale for its inclusion.

| Feature | Description | Rationale |
|---------|-------------|----------|
| `numVariables` | Number of variables in the closed system (from dataset) | Controls complexity; more variables → more possible causal structures |
| `numCorrelations` | Count of 'X correlates with Y' statements in premise | Captures the density of statistical associations |
| `numIndependencies` | Count of all independence statements (simple + conditional) | Independence constraints are key to narrowing causal structures |
| `numCondIndependencies` | Count of specifically conditional independence statements | Conditional independencies are more informative for causal discovery |
| `numPremiseSentences` | Number of sentences in the premise | Proxy for information richness of the premise |
| `totalRelations` | numCorrelations + numIndependencies | Total amount of statistical evidence provided |
| `relationDensity` | numCorrelations / (numVars choose 2) | How dense the correlation graph is; sparser → more independence |
| `independenceRatio` | numIndependencies / totalRelations | Fraction of relations that are independencies vs correlations |
| `avgRelationsPerVar` | totalRelations / numVariables | Average information per variable |
| `premiseWordCount` | Word count of the premise text | Text length feature; longer premises have more complex structures |
| `hypothesisWordCount` | Word count of the hypothesis text | Hypothesis complexity indicator |
| `premiseCharCount` | Character count of the premise text | Complementary length feature |
| `inputWordCount` | Total word count (premise + hypothesis) | Overall text length |
| `premiseCueWords` | Count of causal cue words in premise | Causal language density in premise |
| `hypothesisCueWords` | Count of causal cue words in hypothesis | Causal language density in hypothesis |
| `totalCueWords` | Total causal cue words across premise and hypothesis | Overall causal language presence |
| `numHypoVarsInPremise` | Number of hypothesis variables also in premise | Whether the hypothesis refers to variables with known relations |
| `numHypoVars` | Number of variables mentioned in hypothesis | Hypothesis specificity |
| `hasConditionalIndep` | Binary: premise contains conditional independence statements | Conditional independencies strongly constrain possible DAGs |
| `hasHowever` | Binary: premise contains 'However' | Signals transition from correlations to independence constraints |
| `hypoType_directCause` | Binary: hypothesis type is 'directCause' | Controls for different causal claim types which have different base rates |
| `hypoType_ancestor` | Binary: hypothesis type is 'ancestor' | Controls for different causal claim types which have different base rates |
| `hypoType_descendant` | Binary: hypothesis type is 'descendant' | Controls for different causal claim types which have different base rates |
| `hypoType_collider` | Binary: hypothesis type is 'collider' | Controls for different causal claim types which have different base rates |
| `hypoType_confounder` | Binary: hypothesis type is 'confounder' | Controls for different causal claim types which have different base rates |
| `hypoType_parent` | Binary: hypothesis type is 'parent' | Controls for different causal claim types which have different base rates |
| `hypoType_child` | Binary: hypothesis type is 'child' | Controls for different causal claim types which have different base rates |
| `hypoType_other` | Binary: hypothesis type is 'other' | Controls for different causal claim types which have different base rates |

**Total features**: 28 (excluding label, docId, template metadata)
