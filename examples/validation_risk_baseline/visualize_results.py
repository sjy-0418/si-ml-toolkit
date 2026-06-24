from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from train_signal_integrity_model import (
    FEATURE_COLUMNS,
    Standardizer,
    classification_metrics,
    generate_signal_integrity_dataset,
    sigmoid,
    train_logistic_regression,
)


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "outputs" / "figures"


def svg_text(x: float, y: float, text: str, size: int = 13, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#172033">{escape(text)}</text>'
    )


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#ffffff"/>',
                *body,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    color: str = "#2563eb",
    signed: bool = False,
) -> None:
    width = 860
    height = max(300, 90 + len(labels) * 46)
    left = 220
    right = 52
    top = 64
    row_height = 42
    plot_width = width - left - right
    max_abs = max(abs(value) for value in values) or 1.0
    zero_x = left + (plot_width / 2 if signed else 0)

    body = [
        svg_text(32, 34, title, size=22),
        f'<line x1="{left}" y1="{top - 18}" x2="{left + plot_width}" y2="{top - 18}" stroke="#d7dde8"/>',
    ]

    if signed:
        body.append(
            f'<line x1="{zero_x:.1f}" y1="{top - 26}" x2="{zero_x:.1f}" '
            f'y2="{height - 32}" stroke="#9aa7bb" stroke-dasharray="4 4"/>'
        )

    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * row_height
        body.append(svg_text(32, y + 17, label, size=13))

        if signed:
            bar_len = abs(value) / max_abs * (plot_width / 2 - 8)
            x = zero_x if value >= 0 else zero_x - bar_len
            fill = "#dc2626" if value >= 0 else "#2563eb"
        else:
            bar_len = value / max_abs * plot_width
            x = left
            fill = color

        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_len:.1f}" height="24" '
            f'rx="4" fill="{fill}"/>'
        )
        body.append(svg_text(left + plot_width + 12, y + 17, f"{value:.3f}", size=12))

    write_svg(path, width, height, body)


def line_chart(path: Path, title: str, rows: pd.DataFrame) -> None:
    width = 860
    height = 430
    left = 70
    right = 36
    top = 72
    bottom = 64
    plot_width = width - left - right
    plot_height = height - top - bottom

    def point(threshold: float, metric: float) -> tuple[float, float]:
        x = left + (threshold - 0.05) / 0.90 * plot_width
        y = top + (1.0 - metric) * plot_height
        return x, y

    precision_points = " ".join(
        f"{x:.1f},{y:.1f}"
        for x, y in (point(row.threshold, row.precision) for row in rows.itertuples())
    )
    recall_points = " ".join(
        f"{x:.1f},{y:.1f}"
        for x, y in (point(row.threshold, row.recall) for row in rows.itertuples())
    )

    body = [
        svg_text(32, 36, title, size=22),
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#f8fafc" stroke="#d7dde8"/>',
    ]

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + (1.0 - tick) * plot_height
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e7ebf2"/>')
        body.append(svg_text(left - 12, y + 4, f"{tick:.2f}", size=11, anchor="end"))

    for tick in [0.05, 0.25, 0.50, 0.75, 0.95]:
        x = left + (tick - 0.05) / 0.90 * plot_width
        body.append(svg_text(x, height - 30, f"{tick:.2f}", size=11, anchor="middle"))

    body.extend(
        [
            f'<polyline points="{precision_points}" fill="none" stroke="#2563eb" stroke-width="3"/>',
            f'<polyline points="{recall_points}" fill="none" stroke="#dc2626" stroke-width="3"/>',
            svg_text(width - 220, 40, "Precision", size=13),
            f'<line x1="{width - 280}" y1="36" x2="{width - 232}" y2="36" stroke="#2563eb" stroke-width="3"/>',
            svg_text(width - 100, 40, "Recall", size=13),
            f'<line x1="{width - 156}" y1="36" x2="{width - 112}" y2="36" stroke="#dc2626" stroke-width="3"/>',
            svg_text(left + plot_width / 2, height - 8, "Classification threshold", size=12, anchor="middle"),
            svg_text(18, top + plot_height / 2, "Metric", size=12),
        ]
    )

    write_svg(path, width, height, body)


def scatter_chart(path: Path, title: str, eye_height: np.ndarray, probabilities: np.ndarray) -> None:
    width = 860
    height = 440
    left = 76
    right = 36
    top = 70
    bottom = 64
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_min, x_max = float(eye_height.min()), float(eye_height.max())

    body = [
        svg_text(32, 36, title, size=22),
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#f8fafc" stroke="#d7dde8"/>',
    ]

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + (1.0 - tick) * plot_height
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e7ebf2"/>')
        body.append(svg_text(left - 12, y + 4, f"{tick:.2f}", size=11, anchor="end"))

    for tick in np.linspace(x_min, x_max, 5):
        x = left + (tick - x_min) / (x_max - x_min) * plot_width
        body.append(svg_text(x, height - 30, f"{tick:.0f}", size=11, anchor="middle"))

    order = np.linspace(0, len(eye_height) - 1, 550, dtype=int)
    for idx in order:
        x = left + (eye_height[idx] - x_min) / (x_max - x_min) * plot_width
        y = top + (1.0 - probabilities[idx]) * plot_height
        color = "#dc2626" if probabilities[idx] >= 0.5 else "#2563eb"
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" opacity="0.42"/>')

    body.extend(
        [
            svg_text(left + plot_width / 2, height - 8, "Eye height (mV)", size=12, anchor="middle"),
            svg_text(18, top + plot_height / 2, "Predicted failure risk", size=12),
        ]
    )

    write_svg(path, width, height, body)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_signal_integrity_dataset()
    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["validation_fail"].to_numpy(dtype=float)

    standardizer = Standardizer.fit(x)
    x_scaled = standardizer.transform(x)
    weights, bias, _ = train_logistic_regression(x_scaled, y)
    probabilities = sigmoid(x_scaled @ weights + bias)

    pass_count = int((y == 0).sum())
    fail_count = int((y == 1).sum())
    bar_chart(
        FIGURE_DIR / "class_balance.svg",
        "Validation Class Balance",
        ["pass", "fail"],
        [pass_count, fail_count],
        color="#2563eb",
    )

    feature_weights = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "weight": weights})
        .assign(abs_weight=lambda frame: frame["weight"].abs())
        .sort_values("abs_weight", ascending=True)
    )
    bar_chart(
        FIGURE_DIR / "feature_influence.svg",
        "Feature Influence on Failure Risk",
        feature_weights["feature"].tolist(),
        feature_weights["weight"].tolist(),
        signed=True,
    )

    threshold_rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = classification_metrics(y, (probabilities >= threshold).astype(int))
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )
    line_chart(
        FIGURE_DIR / "precision_recall_threshold.svg",
        "Precision/Recall Threshold Trade-off",
        pd.DataFrame(threshold_rows),
    )

    scatter_chart(
        FIGURE_DIR / "eye_height_risk_scatter.svg",
        "Eye Height vs Predicted Failure Risk",
        df["eye_height_mv"].to_numpy(dtype=float),
        probabilities,
    )

    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()

