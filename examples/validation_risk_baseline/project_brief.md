# Project Brief: Signal Integrity Failure Risk Predictor

## Career Positioning

Target role family:

- Edge AI Engineer
- Applied Machine Learning Engineer
- Computer Vision / Industrial AI Engineer
- ML Systems Engineer with hardware awareness

This project is designed to turn hardware validation experience into a concrete AI/ML portfolio artifact.

## Problem Statement

High-speed interface validation often produces measurable signals before a failure is visible at the product level. A model can help prioritize validation cases that are likely to fail by learning from features such as:

- eye height
- eye width
- skew
- jitter
- impedance error
- supply ripple
- temperature
- interface speed

## Initial Dataset Strategy

The first version uses synthetic data. This is acceptable for a portfolio starter because the objective is to show:

- correct ML framing
- hardware-domain feature design
- reproducible training code
- engineering interpretation

Later versions should replace or augment the synthetic data with public datasets or anonymized lab-style measurements.

## Evaluation Strategy

Primary metrics:

- accuracy
- precision
- recall
- F1 score

For hardware validation, recall is especially important because missed failures can be expensive.

## Suggested Repository Story

The README should emphasize:

> A hardware engineer built an ML workflow that predicts validation risk from signal-integrity features and explains which engineering factors drive the risk.

