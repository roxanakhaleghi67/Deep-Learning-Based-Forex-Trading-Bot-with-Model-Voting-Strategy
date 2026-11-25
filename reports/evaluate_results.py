"""Evaluate the model-voting strategy results stored in ``output.csv``.

Pure standard library: no third-party packages are required, so this runs on a
fresh Python installation. It reports the win rate and total return, computes
the directional accuracy of each model, and writes a Markdown summary plus a
static SVG equity curve into this ``reports`` folder.

Run from the repository root:

    python reports/evaluate_results.py
"""

from __future__ import annotations

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUT_CSV = os.path.join(ROOT, "output.csv")


def load_results(path: str = OUTPUT_CSV) -> tuple[list[str], list[dict]]:
    """Load ``output.csv`` (its first row is a human-readable description)."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = [c.strip() for c in rows[1]]
    records = []
    for row in rows[2:]:
        if not row or not row[0].strip():
            continue
        records.append(dict(zip(header, row)))
    return header, records


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def win_rate_and_profit(records: list[dict]) -> tuple[float, float, int, int]:
    trades = [to_float(r.get("strategy", "")) for r in records]
    trades = [t for t in trades if t is not None]
    wins = sum(1 for t in trades if t > 0)
    total = len(trades)
    win_rate = wins / total if total else 0.0
    return win_rate, float(sum(trades)), wins, total


def directional_accuracy(records: list[dict], model_cols: list[str]) -> dict[str, float]:
    pairs = [(to_float(r.get("real returns", "")), {m: to_float(r.get(m, "")) for m in model_cols})
             for r in records]
    accuracy: dict[str, float] = {}
    for name in model_cols:
        correct = considered = 0
        for actual, preds in pairs:
            if actual is None or actual == 0 or preds.get(name) is None:
                continue
            considered += 1
            if (1 if preds[name] - 0.5 > 0 else -1) == (1 if actual > 0 else -1):
                correct += 1
        accuracy[name] = correct / considered if considered else float("nan")
    return accuracy


def write_svg(points: list[float], path: str) -> None:
    width, height, pad = 900, 420, 40
    low, high = min([*points, 0.0]), max([*points, 0.0])
    span = (high - low) or 1.0
    count = len(points)

    def x_of(i: int) -> float:
        return pad + (width - 2 * pad) * (i / (count - 1) if count > 1 else 0)

    def y_of(v: float) -> float:
        return height - pad - (height - 2 * pad) * ((v - low) / span)

    polyline = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(points))
    zero_y = y_of(0.0)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="white"/>\n'
        f'<line x1="{pad}" y1="{zero_y:.1f}" x2="{width - pad}" y2="{zero_y:.1f}" '
        f'stroke="#999" stroke-dasharray="4"/>\n'
        f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{polyline}"/>\n'
        f'<text x="{pad}" y="24" font-family="sans-serif" font-size="14">'
        f'Cumulative strategy return (model voting)</text>\n'
        f"</svg>\n"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)


def main() -> None:
    header, records = load_results()
    model_cols = [c for c in header if c.lower().startswith("model")]

    win_rate, profit, wins, total = win_rate_and_profit(records)
    accuracy = directional_accuracy(records, model_cols)
    equity, running = [], 0.0
    for record in records:
        value = to_float(record.get("strategy", ""))
        running += value or 0.0
        equity.append(running)

    lines = [
        "# Strategy Evaluation",
        "",
        f"- Rows evaluated: {len(records)}",
        f"- Trades: {total}",
        f"- Win rate: {win_rate:.1%} ({wins}/{total})",
        f"- Total return (sum of strategy returns): {profit:.4f}",
    ]
    if accuracy:
        lines += ["", "## Directional accuracy per model", "", "| Model | Accuracy |", "| --- | --- |"]
        lines += [f"| {name} | {value:.1%} |" for name, value in accuracy.items()]

    summary = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "results_summary.md"), "w", encoding="utf-8") as handle:
        handle.write(summary)

    if equity:
        write_svg(equity, os.path.join(HERE, "equity_curve.svg"))

    print(summary)


if __name__ == "__main__":
    main()
