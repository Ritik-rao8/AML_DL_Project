"""Historical conservative rule baseline for comparison."""
import re

def ruleBasedPredict(inputText: str) -> int:
    """
    Rule-based prediction: determine if the causal hypothesis follows from
    the premise based on pattern matching of statistical relations.
    
    The key insight is that in Corr2Cause:
    - Label 1 (valid): the causal claim IS entailed by the statistical relations
    - Label 0 (invalid): the causal claim is NOT entailed
    
    Rules:
    1. If variables are independent, no direct causal link exists → reject direct cause claims
    2. For collider claims: need convergent patterns (A→C←B)
    3. For confounder claims: need divergent patterns (A←C→B)
    4. Direct cause (parent): correlation must exist AND certain independence patterns
    5. Conservative default: predict 0 (most hypotheses are not entailed)
    """
    if "Hypothesis:" not in inputText:
        return 0
    
    parts = inputText.split("Hypothesis:")
    premise = parts[0].replace("Premise:", "").strip().lower()
    hypothesis = parts[1].strip().lower()
    
    # Extract mentioned variables in hypothesis
    hypoVars = re.findall(r'\b([a-f])\b', hypothesis)
    if len(hypoVars) < 2:
        return 0
    
    var1, var2 = hypoVars[0], hypoVars[1]
    
    # Check if the two main hypothesis variables correlate
    varsCorrelate = (
        f"{var1} correlates with {var2}" in premise or 
        f"{var2} correlates with {var1}" in premise
    )
    
    # Check if they are independent
    varsIndependent = (
        f"{var1} is independent of {var2}" in premise or
        f"{var2} is independent of {var1}" in premise or
        f"{var1} and {var2} are independent" in premise or
        f"{var2} and {var1} are independent" in premise
    )
    
    # Count how many conditional independencies involve these variables
    condIndepCount = premise.count(f"{var1} and {var2} are independent given")
    condIndepCount += premise.count(f"{var2} and {var1} are independent given")
    
    # Number of variables in the system
    allVars = set(re.findall(r'\b([a-f])\b', premise))
    numVars = len(allVars)
    
    # ── Direct cause (parent) claims ──
    if "directly causes" in hypothesis or "direct cause" in hypothesis:
        # If variables are unconditionally independent, can't be direct cause
        if varsIndependent:
            return 0
        # If they correlate and there are conditional independencies with
        # other variables, it might be a valid direct cause in a small system
        if varsCorrelate and numVars == 2:
            # With only 2 variables and correlation, direct cause is possible
            # but not guaranteed (could be reverse causation)
            return 0  # Conservative: can't determine direction from correlation alone
        if varsCorrelate and condIndepCount == 0 and numVars <= 3:
            return 0  # Not enough independence info to confirm
        return 0  # Default conservative
    
    # ── Collider claims ──
    if "collider" in hypothesis or "common effect" in hypothesis:
        if varsIndependent:
            # For a collider structure (A→C←B), A and B are marginally independent
            # but become dependent when conditioning on C
            # Check if they become dependent given some variable
            otherVars = allVars - {var1, var2}
            for v in otherVars:
                condDep = (
                    f"{var1} correlates with {var2}" not in premise and
                    not any(f"{var1} and {var2} are independent given" in s 
                           for s in [premise])
                )
            return 0  # Conservative
        return 0
    
    # ── Confounder claims ──
    if "confounder" in hypothesis or "common cause" in hypothesis:
        if varsCorrelate:
            # A confounder means A←C→B: A and B correlate but become
            # independent when conditioning on C
            otherVars = allVars - {var1, var2}
            for v in otherVars:
                givenV = f"{var1} and {var2} are independent given {v}"
                givenV2 = f"{var2} and {var1} are independent given {v}"
                if givenV in premise or givenV2 in premise:
                    # They become independent given a third variable → suggests confounder
                    return 1
        return 0
    
    # ── Ancestor / descendant claims ──
    if "ancestor" in hypothesis or "not a direct one" in hypothesis:
        if varsCorrelate:
            return 0  # Can't confirm indirect cause from correlations alone
        return 0
    
    if "descendant" in hypothesis:
        if varsCorrelate:
            return 0
        return 0
    
    # Default: predict 0 (conservative)
    return 0
