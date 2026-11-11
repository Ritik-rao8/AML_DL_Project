"""The shared 28-feature Phase 1 extractor; no output-directory side effects."""
import re
import pandas as pd
from src.config import COL_INPUT, COL_LABEL, COL_NUM_VARS, COL_TEMPLATE
from src import data as loader

CAUSAL_CUE_WORDS = [
    "cause", "causes", "caused", "causing",
    "effect", "affects", "affect",
    "leads", "lead",
    "results", "result",
    "due", "because",
    "therefore", "thus", "hence",
    "implies", "imply",
    "determines", "determine",
    "influences", "influence",
    "produces", "produce",
    "directly",
    "indirectly",
    "ancestor",
    "descendant",
    "collider",
    "confounder",
    "mediator",
]

CORRELATION_PATTERN = re.compile(r'\b([A-F])\s+correlates?\s+with\s+([A-F])\b', re.IGNORECASE)

INDEPENDENCE_PATTERN = re.compile(r'\b([A-F])\s+(?:is\s+)?independent\s+(?:of|from)\s+([A-F])\b', re.IGNORECASE)

CONDITIONAL_INDEP_PATTERN = re.compile(
    r'\b([A-F])\s+and\s+([A-F])\s+are\s+independent\s+given\s+((?:[A-F](?:\s+and\s+)?)+)',
    re.IGNORECASE
)

HYPOTHESIS_TYPES = {
    "directCause": ["directly causes", "direct cause"],
    "ancestor": ["ancestor", "not a direct one", "indirect cause"],
    "descendant": ["descendant"],
    "collider": ["collider", "common effect"],
    "confounder": ["confounder", "common cause"],
    "parent": ["directly causes"],
    "child": ["directly caused by"],
}

def extractVariables(text: str) -> set:
    """Extract all variable names (A–F) mentioned in the text."""
    return set(re.findall(r'\b([A-F])\b', text))

def countCorrelations(premiseText: str) -> int:
    """Count the number of correlation statements in the premise."""
    return len(CORRELATION_PATTERN.findall(premiseText))

def countIndependencies(premiseText: str) -> int:
    """Count the number of independence statements in the premise."""
    simple = len(INDEPENDENCE_PATTERN.findall(premiseText))
    conditional = len(CONDITIONAL_INDEP_PATTERN.findall(premiseText))
    return simple + conditional

def countConditionalIndependencies(premiseText: str) -> int:
    """Count specifically conditional independence statements."""
    return len(CONDITIONAL_INDEP_PATTERN.findall(premiseText))

def countCausalCueWords(text: str) -> int:
    """Count occurrences of causal cue words in the text."""
    textLower = text.lower()
    return sum(1 for cue in CAUSAL_CUE_WORDS if cue in textLower)

def classifyHypothesisType(hypothesisText: str) -> str:
    """Classify the hypothesis into a type based on keywords."""
    hypoLower = hypothesisText.lower()
    for hType, keywords in HYPOTHESIS_TYPES.items():
        if any(kw in hypoLower for kw in keywords):
            return hType
    return "other"

def computeRelationDensity(numCorrelations: int, numVars: int) -> float:
    """
    Compute the density of correlations relative to possible pairs.
    For n variables, max pairs = n*(n-1)/2.
    """
    if numVars < 2:
        return 0.0
    maxPairs = numVars * (numVars - 1) / 2
    return numCorrelations / maxPairs if maxPairs > 0 else 0.0

def computeIndependenceRatio(numIndependencies: int, numCorrelations: int) -> float:
    """Ratio of independence statements to correlation statements."""
    total = numIndependencies + numCorrelations
    return numIndependencies / total if total > 0 else 0.0

def extractFeaturesFromRow(row: pd.Series) -> dict:
    """
    Extract all features from a single data row.
    
    Returns a dict of feature_name -> feature_value.
    """
    inputText = row[COL_INPUT]
    numVars = int(row[COL_NUM_VARS])
    
    premiseText = row.get("premise", loader.extractPremise(inputText))
    hypothesisText = row.get("hypothesis", loader.extractHypothesis(inputText))

    # ── Structural features ──
    variables = extractVariables(premiseText)
    numCorrelations = countCorrelations(premiseText)
    numIndependencies = countIndependencies(premiseText)
    numCondIndependencies = countConditionalIndependencies(premiseText)
    
    # ── Hypothesis features ──
    hypoVariables = extractVariables(hypothesisText)
    hypoType = classifyHypothesisType(hypothesisText)
    
    # ── Text statistics ──
    premiseWordCount = len(premiseText.split())
    hypothesisWordCount = len(hypothesisText.split())
    premiseCharCount = len(premiseText)
    
    # ── Cue words ──
    premiseCueWords = countCausalCueWords(premiseText)
    hypothesisCueWords = countCausalCueWords(hypothesisText)
    
    # ── Derived features ──
    relationDensity = computeRelationDensity(numCorrelations, numVars)
    indepRatio = computeIndependenceRatio(numIndependencies, numCorrelations)
    
    # Number of premise sentences (approximate by splitting on '. ')
    numPremiseSentences = premiseText.count('. ') + 1
    
    # Whether hypothesis mentions variables from the premise
    hypoVarsInPremise = len(hypoVariables & variables)
    
    # Average relations per variable
    totalRelations = numCorrelations + numIndependencies
    avgRelationsPerVar = totalRelations / numVars if numVars > 0 else 0
    
    features = {
        # Structural
        "numVariables": numVars,
        "numCorrelations": numCorrelations,
        "numIndependencies": numIndependencies,
        "numCondIndependencies": numCondIndependencies,
        "numPremiseSentences": numPremiseSentences,
        "totalRelations": totalRelations,
        
        # Density / ratio
        "relationDensity": round(relationDensity, 4),
        "independenceRatio": round(indepRatio, 4),
        "avgRelationsPerVar": round(avgRelationsPerVar, 4),
        
        # Text stats
        "premiseWordCount": premiseWordCount,
        "hypothesisWordCount": hypothesisWordCount,
        "premiseCharCount": premiseCharCount,
        "inputWordCount": premiseWordCount + hypothesisWordCount,
        
        # Cue words
        "premiseCueWords": premiseCueWords,
        "hypothesisCueWords": hypothesisCueWords,
        "totalCueWords": premiseCueWords + hypothesisCueWords,
        
        # Hypothesis type (one-hot encoded below)
        "hypoType": hypoType,
        
        # Variable overlap
        "numHypoVarsInPremise": hypoVarsInPremise,
        "numHypoVars": len(hypoVariables),
        
        # Has conditional independence (binary)
        "hasConditionalIndep": int(numCondIndependencies > 0),
        
        # Premise mentions "however" (signals independence constraints)
        "hasHowever": int("however" in premiseText.lower()),
    }
    
    return features

def oneHotEncodeHypoType(featuresDf: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the hypoType column."""
    # Get all possible types
    allTypes = ["directCause", "ancestor", "descendant", "collider", "confounder", "parent", "child", "other"]
    
    for hType in allTypes:
        featuresDf[f"hypoType_{hType}"] = (featuresDf["hypoType"] == hType).astype(int)
    
    featuresDf = featuresDf.drop(columns=["hypoType"])
    return featuresDf

def extractAllFeatures(df: pd.DataFrame, splitName: str) -> pd.DataFrame:
    """
    Extract features for all rows in a dataframe.
    Returns a DataFrame with the features, label, and metadata columns.
    """
    print(f"  Extracting features for {splitName} ({len(df):,} rows)...")
    
    featuresList = []
    for idx, row in df.iterrows():
        feats = extractFeaturesFromRow(row)
        feats["label"] = int(row[COL_LABEL])
        feats["docId"] = row.get("docId", f"{splitName}_{idx}")
        feats["template"] = row[COL_TEMPLATE]
        featuresList.append(feats)
    
    featuresDf = pd.DataFrame(featuresList)
    featuresDf = oneHotEncodeHypoType(featuresDf)
    
    print(f"    → {featuresDf.shape[1] - 3} numeric features extracted")
    return featuresDf
