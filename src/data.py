"""Shared Corr2Cause parsing and content identifiers."""
import hashlib
import pandas as pd
from src.config import COL_INPUT, COL_LABEL, COL_NUM_VARS, COL_TEMPLATE

def generateDocId(text: str, split: str, index: int) -> str:
    """
    Generate a stable, unique document identifier from the text content.
    Uses the full SHA-256 content hash, independent of split and row order.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def loadAndProcess(csvPath: str, splitName: str) -> pd.DataFrame:
    """
    Load a raw CSV, assign document and sentence IDs, and validate schema.
    
    Each row in Corr2Cause contains one premise–hypothesis pair, which we 
    treat as one 'document'. The premise may contain multiple sentences 
    describing correlations/independencies, and we assign sentence IDs 
    to each within the premise.
    """
    df = pd.read_csv(csvPath)
    
    # Validate expected columns
    expectedCols = {COL_INPUT, COL_LABEL, COL_NUM_VARS, COL_TEMPLATE}
    actualCols = set(df.columns)
    assert expectedCols.issubset(actualCols), \
        f"Missing columns: {expectedCols - actualCols}"
    
    # Assign stable document IDs
    df["docId"] = [
        generateDocId(row[COL_INPUT], splitName, idx) 
        for idx, row in df.iterrows()
    ]
    
    # Assign split name
    df["split"] = splitName
    
    # Parse premise and hypothesis from the input text
    df["premise"] = df[COL_INPUT].apply(extractPremise)
    df["hypothesis"] = df[COL_INPUT].apply(extractHypothesis)
    
    # Assign sentence IDs within each document (premise sentences)
    df["premiseSentences"] = df["premise"].apply(splitIntoSentences)
    df["numPremiseSentences"] = df["premiseSentences"].apply(len)
    
    # Ensure label is integer
    df[COL_LABEL] = df[COL_LABEL].astype(int)
    df[COL_NUM_VARS] = df[COL_NUM_VARS].astype(int)
    
    return df

def extractPremise(inputText: str) -> str:
    """Extract the premise part from the input text."""
    if "Hypothesis:" in inputText:
        parts = inputText.split("Hypothesis:")
        premise = parts[0].replace("Premise:", "").strip()
        return premise
    return inputText.replace("Premise:", "").strip()

def extractHypothesis(inputText: str) -> str:
    """Extract the hypothesis part from the input text."""
    if "Hypothesis:" in inputText:
        parts = inputText.split("Hypothesis:")
        return parts[1].strip()
    return ""

def splitIntoSentences(premiseText: str) -> list:
    """
    Split a premise into individual sentences describing relations.
    We split on period-space patterns but keep the content meaningful.
    """
    # Split on '. ' but keep sentences clean
    sentences = []
    current = ""
    for char in premiseText:
        current += char
        if char == '.' and len(current.strip()) > 5:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences
