# Signal Integrity Failure Risk Predictor

## Dataset

- Samples: 2,500 synthetic validation cases
- Validation fail rate: 58.6%

## Model

- Logistic regression implemented with NumPy
- Standardized hardware validation features
- Final training loss: 0.3863

## Test Metrics

- Accuracy: 0.844
- Precision: 0.890
- Recall: 0.856
- F1 score: 0.873
- False negatives: 45

## High-Recall Screening Threshold

- Threshold: 0.43
- Precision: 0.870
- Recall: 0.901
- F1 score: 0.885
- False negatives: 31

## Feature Influence

| feature | weight |
| --- | ---: |
| eye_height_mv | -0.7859 |
| interface_speed_gbps | 0.7052 |
| eye_width_ui | -0.5617 |
| jitter_rms_ps | 0.4268 |
| skew_ps | 0.3851 |
| supply_ripple_mv | 0.2265 |
| impedance_error_ohm | 0.2189 |
| temperature_c | 0.1871 |

## Engineering Interpretation

A positive feature weight increases predicted validation-failure risk.
A negative feature weight reduces predicted validation-failure risk.
For hardware validation, recall should be watched closely because missed failures are more costly than extra debug candidates.

## Next Steps

1. Add a public or anonymized real-world validation dataset.
2. Compare with tree-based models after adding scikit-learn.
3. Add threshold tuning for high-recall validation screening.
4. Deploy the model behind a small FastAPI service.