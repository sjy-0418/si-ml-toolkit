# EyeScope Auto Batch Validation Report

## Summary

- Captures evaluated: 4
- Passed: 1
- Failed: 3
- Pass rate: 25.0%

## Corner Matrix

| Capture ID | Corner | Overall | Eye height (V) | Eye width (UI) | Jitter (UI) | Failed checks |
| --- | --- | :---: | ---: | ---: | ---: | --- |
| nominal_25c | Nominal | PASS | 0.960 | 0.581 | 0.0079 | - |
| low_voltage_margin | Low voltage margin | FAIL | 0.960 | 0.581 | 0.0079 | eye_height |
| high_temp_clock | High temperature clock | FAIL | 0.960 | 0.581 | 0.0079 | jitter |
| tight_timing_mask | Tight timing mask | FAIL | 0.960 | 0.581 | 0.0079 | eye_width |

## Debug Focus

- `low_voltage_margin` (Low voltage margin): eye_height. Low eye height may indicate noise margin degradation, attenuation, or a power integrity issue.
- `high_temp_clock` (High temperature clock): jitter. High jitter may indicate clock instability, crosstalk, or a signal integrity issue.
- `tight_timing_mask` (Tight timing mask): eye_width. Narrow eye width may indicate a timing margin issue, inter-symbol interference, or channel loss.

## Validation Interpretation

A batch report is useful for validation teams because it turns repeated lab captures into a reviewable regression matrix. The goal is not only to mark pass/fail, but to identify which corners deserve debug attention first.
