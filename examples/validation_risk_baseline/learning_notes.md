# Learning Notes: Validation Risk Baseline

This project is not only about producing a model. It is a study exercise for learning how to translate a hardware validation problem into an ML workflow.

## 1. Problem Framing

The ML task is binary classification:

- `0`: validation case passes
- `1`: validation case fails

In a real SI validation workflow, the model would not replace engineering judgment. It would help prioritize which cases deserve earlier review.

## 2. Why Logistic Regression First?

Logistic regression is a good first baseline because it is simple and interpretable.

The model learns one weight per input feature. A positive weight increases predicted failure risk. A negative weight reduces predicted failure risk.

That makes it easier to connect ML output back to engineering intuition:

- lower eye height should increase failure risk
- lower eye width should increase failure risk
- higher skew should increase failure risk
- higher jitter should increase failure risk
- higher supply ripple should increase failure risk

## 3. Why Feature Scaling Matters

The input features use different units:

- Gbps
- millivolts
- UI
- picoseconds
- ohms
- Celsius

Without scaling, features with larger numeric ranges can dominate training even when they are not physically more important.

Standardization converts each feature into:

```text
scaled_value = (value - mean) / standard_deviation
```

This makes gradient descent behave more predictably.

## 4. Precision vs Recall

Precision answers:

> Of the cases predicted as fail, how many really failed?

Recall answers:

> Of all real fail cases, how many did the model catch?

For validation screening, recall is often more important than raw accuracy because missed failures can become expensive debug problems later.

## 5. Threshold Tuning

Logistic regression outputs a probability. The default decision threshold is usually `0.50`.

For hardware validation, that may not be the best operating point. If missing a real failure is costly, we can lower the threshold to catch more risky cases.

Trade-off:

- lower threshold: higher recall, more false alarms
- higher threshold: higher precision, more missed failures

This is why the example reports both default metrics and a high-recall screening threshold.

## 6. How To Explain This In An Interview

A strong explanation:

> I started with a simple logistic-regression baseline because I wanted the model behavior to be interpretable. I used SI-inspired features such as eye height, eye width, skew, jitter, impedance error, supply ripple, and temperature. Then I tuned the classification threshold for a high-recall validation workflow, because in hardware validation missing a failure can be more expensive than reviewing extra candidates.

## 7. What To Study Next

Recommended next topics:

1. Confusion matrix
2. ROC curve and PR curve
3. Scikit-learn logistic regression
4. Random forest and gradient boosting baselines
5. Model calibration
6. Real eye-diagram or S-parameter feature extraction

