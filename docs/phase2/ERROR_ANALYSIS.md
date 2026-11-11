# Held-out error analysis

Examples below are for reporting only; they did not select models or thresholds. Full text is retained in the prediction CSVs.

## False positives

- ID: 41c006b05015f95062207ba6d36378d311125f4ec97dda738398873a0dcf6c68; score 0.6338
  Hypothesis: There exists at least one collider (i.e., common effect) of A and B.
  Premise excerpt: Suppose there is a closed system of 2 variables, A and B. All the statistical relations among these 2 variables are as follows: A correlates with B.

- ID: b3c8f1ce87979094c0228d2d15a6f2207dd1f80bbd4c64a1c978ef98e92d3fad; score 0.6293
  Hypothesis: There exists at least one collider (i.e., common effect) of B and C.
  Premise excerpt: Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among these 3 variables are as follows: A correlates with C. B correlates with C. However, A is independent of B.

- ID: a8ff3933b7396ae8a9635748f921d5da9b0aaee3994911b82d0f609d4aec8df3; score 0.6285
  Hypothesis: There exists at least one collider (i.e., common effect) of A and C.
  Premise excerpt: Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among these 3 variables are as follows: A correlates with C. B correlates with C. However, A is independent of B.

## False negatives

- ID: 86b82cfaf1d172d40e22031b79d8685ac1f62c86b1f51695bfb220659aa259a7; score 0.0484
  Hypothesis: A directly causes C.
  Premise excerpt: Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among these 3 variables are as follows: A correlates with C. B correlates with C. However, A is independent of B.

- ID: ec70511ea3cb856305a0c9f75e0046a8e330a592a7491330e547c0e7b5193fe6; score 0.0342
  Hypothesis: B directly causes C.
  Premise excerpt: Suppose there is a closed system of 3 variables, A, B and C. All the statistical relations among these 3 variables are as follows: A correlates with C. B correlates with C. However, A is independent of B.

- ID: fe7a1456ce8ced9796159b64cc3f3d73ab8836cdfd56d24deae498e9b47d3a1d; score 0.0605
  Hypothesis: A directly causes C.
  Premise excerpt: Suppose there is a closed system of 5 variables, A, B, C, D and E. All the statistical relations among these 5 variables are as follows: A correlates with C. A correlates with D. A correlates with E. B correlates with C. B correlates with D. B correlates with E. C correlates with D. C correlates with E. D correlates with E. However, A is independent of B. A and D are independent given B and C. A and D are independent given B, C and E. B and E are independent given A and C. B and E are independen
