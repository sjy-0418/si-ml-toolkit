from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


FEATURE_COLUMNS = [
    "interface_speed_gbps",
    "eye_height_mv",
    "eye_width_ui",
    "skew_ps",
    "jitter_rms_ps",
    "impedance_error_ohm",
    "supply_ripple_mv",
    "temperature_c",
]


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def generate_signal_integrity_dataset(
    n_samples: int = 2500, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    interface_speed_gbps = rng.choice(
        [3.2, 4.8, 6.4, 8.0, 10.0, 12.0], size=n_samples
    )
    temperature_c = rng.normal(55, 18, size=n_samples).clip(10, 105)
    skew_ps = rng.gamma(shape=2.4, scale=12.0, size=n_samples).clip(0, 130)
    jitter_rms_ps = rng.normal(11 + interface_speed_gbps * 0.9, 3.2, size=n_samples)
    jitter_rms_ps = jitter_rms_ps.clip(3, 38)
    impedance_error_ohm = np.abs(rng.normal(0, 4.5, size=n_samples)).clip(0, 22)
    supply_ripple_mv = rng.normal(18 + interface_speed_gbps * 1.1, 7.5, size=n_samples)
    supply_ripple_mv = supply_ripple_mv.clip(2, 70)

    eye_height_mv = (
        255
        - interface_speed_gbps * 8.5
        - jitter_rms_ps * 2.1
        - supply_ripple_mv * 0.55
        - temperature_c * 0.18
        + rng.normal(0, 16, size=n_samples)
    ).clip(35, 260)

    eye_width_ui = (
        0.76
        - interface_speed_gbps * 0.018
        - skew_ps * 0.0024
        - jitter_rms_ps * 0.006
        + rng.normal(0, 0.045, size=n_samples)
    ).clip(0.08, 0.82)

    risk_score = (
        0.8
        + interface_speed_gbps * 0.24
        + skew_ps * 0.026
        + jitter_rms_ps * 0.095
        + impedance_error_ohm * 0.10
        + supply_ripple_mv * 0.031
        + temperature_c * 0.010
        - eye_height_mv * 0.025
        - eye_width_ui * 5.8
        + rng.normal(0, 0.45, size=n_samples)
    )

    fail_probability = sigmoid(risk_score)
    validation_fail = rng.binomial(1, fail_probability)

    return pd.DataFrame(
        {
            "interface_speed_gbps": interface_speed_gbps,
            "eye_height_mv": eye_height_mv,
            "eye_width_ui": eye_width_ui,
            "skew_ps": skew_ps,
            "jitter_rms_ps": jitter_rms_ps,
            "impedance_error_ohm": impedance_error_ohm,
            "supply_ripple_mv": supply_ripple_mv,
            "temperature_c": temperature_c,
            "validation_fail": validation_fail,
        }
    )


def train_test_split(
    x: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    test_size = int(len(x) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 0.08,
    epochs: int = 2200,
    l2: float = 0.002,
) -> tuple[np.ndarray, float, list[float]]:
    weights = np.zeros(x_train.shape[1])
    bias = 0.0
    losses: list[float] = []

    for _ in range(epochs):
        logits = x_train @ weights + bias
        probabilities = sigmoid(logits)
        error = probabilities - y_train

        grad_w = (x_train.T @ error) / len(x_train) + l2 * weights
        grad_b = float(error.mean())

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        loss = -np.mean(
            y_train * np.log(probabilities + 1e-9)
            + (1 - y_train) * np.log(1 - probabilities + 1e-9)
        )
        losses.append(float(loss))

    return weights, bias, losses


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def choose_threshold_for_recall(
    y_true: np.ndarray, probabilities: np.ndarray, target_recall: float = 0.90
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None

    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = classification_metrics(y_true, (probabilities >= threshold).astype(int))
        if metrics["recall"] >= target_recall and (
            best_metrics is None or metrics["precision"] >= best_metrics["precision"]
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        best_metrics = classification_metrics(
            y_true, (probabilities >= best_threshold).astype(int)
        )

    return best_threshold, best_metrics


def write_report(
    metrics: dict[str, float],
    high_recall_metrics: dict[str, float],
    high_recall_threshold: float,
    feature_weights: pd.DataFrame,
    fail_rate: float,
    final_loss: float,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    feature_table = ["| feature | weight |", "| --- | ---: |"]
    for row in feature_weights.itertuples(index=False):
        feature_table.append(f"| {row.feature} | {row.weight:.4f} |")

    report = [
        "# Signal Integrity Failure Risk Predictor",
        "",
        "## Dataset",
        "",
        "- Samples: 2,500 synthetic validation cases",
        f"- Validation fail rate: {fail_rate:.1%}",
        "",
        "## Model",
        "",
        "- Logistic regression implemented with NumPy",
        "- Standardized hardware validation features",
        f"- Final training loss: {final_loss:.4f}",
        "",
        "## Test Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.3f}",
        f"- Precision: {metrics['precision']:.3f}",
        f"- Recall: {metrics['recall']:.3f}",
        f"- F1 score: {metrics['f1']:.3f}",
        f"- False negatives: {int(metrics['false_negative'])}",
        "",
        "## High-Recall Screening Threshold",
        "",
        f"- Threshold: {high_recall_threshold:.2f}",
        f"- Precision: {high_recall_metrics['precision']:.3f}",
        f"- Recall: {high_recall_metrics['recall']:.3f}",
        f"- F1 score: {high_recall_metrics['f1']:.3f}",
        f"- False negatives: {int(high_recall_metrics['false_negative'])}",
        "",
        "## Feature Influence",
        "",
        "\n".join(feature_table),
        "",
        "## Engineering Interpretation",
        "",
        "A positive feature weight increases predicted validation-failure risk.",
        "A negative feature weight reduces predicted validation-failure risk.",
        "For hardware validation, recall should be watched closely because missed failures are more costly than extra debug candidates.",
        "",
        "## Next Steps",
        "",
        "1. Add a public or anonymized real-world validation dataset.",
        "2. Compare with tree-based models after adding scikit-learn.",
        "3. Add threshold tuning for high-recall validation screening.",
        "4. Deploy the model behind a small FastAPI service.",
    ]
    (OUTPUT_DIR / "model_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = generate_signal_integrity_dataset()
    df.to_csv(OUTPUT_DIR / "signal_integrity_dataset.csv", index=False)

    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["validation_fail"].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    standardizer = Standardizer.fit(x_train)
    x_train_scaled = standardizer.transform(x_train)
    x_test_scaled = standardizer.transform(x_test)

    weights, bias, losses = train_logistic_regression(x_train_scaled, y_train)
    probabilities = sigmoid(x_test_scaled @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)
    metrics = classification_metrics(y_test, predictions)
    high_recall_threshold, high_recall_metrics = choose_threshold_for_recall(
        y_test, probabilities
    )
    high_recall_predictions = (probabilities >= high_recall_threshold).astype(int)

    prediction_df = pd.DataFrame(
        {
            "actual_validation_fail": y_test.astype(int),
            "predicted_validation_fail": predictions,
            "high_recall_prediction": high_recall_predictions,
            "failure_probability": probabilities,
        }
    )
    prediction_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    feature_weights = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "weight": weights})
        .assign(abs_weight=lambda frame: frame["weight"].abs())
        .sort_values("abs_weight", ascending=False)
        .drop(columns=["abs_weight"])
    )
    write_report(
        metrics,
        high_recall_metrics,
        high_recall_threshold,
        feature_weights,
        fail_rate=float(y.mean()),
        final_loss=losses[-1],
    )

    print("Training complete")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 score:  {metrics['f1']:.3f}")
    print(f"High-recall threshold: {high_recall_threshold:.2f}")
    print(f"High-recall recall:    {high_recall_metrics['recall']:.3f}")
    print(f"Report:    {OUTPUT_DIR / 'model_report.md'}")


if __name__ == "__main__":
    main()

