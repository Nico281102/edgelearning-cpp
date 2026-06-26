#!/usr/bin/env python3
"""Generate dependency-free SVG plots for the EL_C_vsCpp STM32N6 sweep."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import math
import re
from pathlib import Path


VARIANTS = ("legacy_c", "cpp_direct_c_backend", "cpp_m55", "cpp_generic", "rltools_generic")
LABELS = {
    "legacy_c": "C M55",
    "cpp_direct_c_backend": "C++ direct C backend",
    "cpp_m55": "C++ M55",
    "cpp_generic": "C++ generic",
    "rltools_generic": "RLTools generic",
}
COLORS = {
    "legacy_c": "#2f2f2f",
    "cpp_direct_c_backend": "#177245",
    "cpp_m55": "#1f62b2",
    "cpp_generic": "#b24a1f",
    "rltools_generic": "#7a3db8",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    today = dt.date.today().isoformat()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Sweep CSV produced by report_sweep_n6.py. Defaults to the latest 10-seed CSV.",
    )
    parser.add_argument("--convergence-config", default="32x32")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "results")
    parser.add_argument("--date", default=today)
    parser.add_argument("--output-tag", default=None)
    return parser.parse_args()


def latest_summary_csv(results_dir: Path) -> Path:
    matches = sorted(
        results_dir.glob("stm32n6_sweep_*_10seed.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise SystemExit("No summary CSV found; run report_sweep_n6.py first.")
    return matches[-1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_int(value: object, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(str(value))


def parse_kv_line(line: str) -> tuple[str, dict[str, str]]:
    parts = line.strip().split()
    if not parts:
        return "", {}
    values: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return parts[0], values


def normalize_config_label(raw: str) -> str:
    parts = raw.split("-")
    if len(parts) >= 3:
        return parts[1]
    return raw.replace("32-", "").replace("-1", "")


def infer_output_tag(summary_csv: Path, explicit_tag: str | None) -> str:
    if explicit_tag is not None:
        return explicit_tag
    match = re.search(r"_input(\d+)(?:_|$)", summary_csv.stem)
    return f"_input{match.group(1)}" if match else ""


def write_speedup_csv(path: Path, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        legacy = to_int(row.get("legacy_c_cycles_avg"))
        for variant in VARIANTS:
            cycles = to_int(row.get(f"{variant}_cycles_avg"))
            speedup = (legacy / cycles) if cycles else 0.0
            out.append(
                {
                    "config": row["config"],
                    "params": to_int(row.get("params")),
                    "variant": variant,
                    "cycles_avg": cycles,
                    "speedup_over_legacy": speedup,
                    "cycles_over_legacy": (cycles / legacy) if legacy else 0.0,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "params",
                "variant",
                "cycles_avg",
                "speedup_over_legacy",
                "cycles_over_legacy",
            ],
        )
        writer.writeheader()
        writer.writerows(out)
    return out


def read_trace_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    traces: list[dict[str, object]] = []
    seen_logs: set[Path] = set()
    for summary in summary_rows:
        log_path = Path(summary.get("log_path", ""))
        if not log_path.exists() or log_path in seen_logs:
            continue
        seen_logs.add(log_path)
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            kind, values = parse_kv_line(line)
            if kind != "TRACE":
                continue
            mse_e9 = to_int(values.get("minibatch_mse_e-9"))
            traces.append(
                {
                    "config": normalize_config_label(values.get("config", "")),
                    "variant": values.get("variant", ""),
                    "seed": to_int(values.get("seed")),
                    "step": to_int(values.get("step")),
                    "sample_passes": to_int(values.get("sample_passes")),
                    "minibatch_mse": mse_e9 / 1_000_000_000.0,
                }
            )
    return traces


def write_convergence_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["config", "variant", "seed", "step", "sample_passes", "minibatch_mse"],
        )
        writer.writeheader()
        writer.writerows(rows)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202020}.axis{stroke:#222;stroke-width:1.2}.grid{stroke:#dddddd;stroke-width:1}.label{font-size:13px}.title{font-size:20px;font-weight:700}.tick{font-size:11px}.legend{font-size:12px}</style>',
    ]


def polyline(points: list[tuple[float, float]], color: str) -> str:
    data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{data}" fill="none" stroke="{color}" stroke-width="2.4"/>'


def circle(x: float, y: float, color: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}"/>'


def draw_legend(lines: list[str], x: int, y: int, variants: tuple[str, ...]) -> None:
    for i, variant in enumerate(variants):
        yy = y + i * 20
        color = COLORS[variant]
        lines.append(f'<line x1="{x}" y1="{yy}" x2="{x + 24}" y2="{yy}" stroke="{color}" stroke-width="2.4"/>')
        lines.append(f'<text class="legend" x="{x + 32}" y="{yy + 4}">{html.escape(LABELS[variant])}</text>')


def write_speedup_svg(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 1100, 620
    left, right, top, bottom = 82, 235, 62, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    params = sorted({int(row["params"]) for row in rows})
    configs_by_params = {int(row["params"]): str(row["config"]) for row in rows}
    speeds = [float(row["speedup_over_legacy"]) for row in rows if float(row["speedup_over_legacy"]) > 0]
    x_min, x_max = min(params), max(params)
    y_min, y_max = 0.0, max(1.0, max(speeds) * 1.12)

    def sx(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min if x_max != x_min else 1.0)) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - ((value - y_min) / (y_max - y_min))) * plot_h

    lines = svg_header(width, height)
    lines.append('<text class="title" x="82" y="34">Speedup over legacy C baseline</text>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = sy(value)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:.2f}x</text>')
    for param in params:
        x = sx(param)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>')
        lines.append(f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 20}" text-anchor="middle">{html.escape(configs_by_params[param])}</text>')
        lines.append(f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 36}" text-anchor="middle">{param}</text>')
    by_variant: dict[str, list[dict[str, object]]] = {variant: [] for variant in VARIANTS}
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    for variant in VARIANTS:
        series = sorted(by_variant[variant], key=lambda row: int(row["params"]))
        points = [(sx(float(row["params"])), sy(float(row["speedup_over_legacy"]))) for row in series]
        lines.append(polyline(points, COLORS[variant]))
        for x, y in points:
            lines.append(circle(x, y, COLORS[variant]))
    lines.append(f'<text class="label" x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle">Network config and parameter count</text>')
    lines.append(f'<text class="label" transform="translate(22 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">legacy C cycles / variant cycles</text>')
    draw_legend(lines, width - right + 34, top + 8, VARIANTS)
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_convergence_svg(path: Path, rows: list[dict[str, object]], config: str) -> None:
    selected = [row for row in rows if row["config"] == config and float(row["minibatch_mse"]) > 0]
    if not selected:
        selected = [row for row in rows if float(row["minibatch_mse"]) > 0]
        config = str(selected[0]["config"]) if selected else config
    width, height = 1100, 620
    left, right, top, bottom = 92, 235, 62, 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    if not selected:
        lines = svg_header(width, height)
        lines.append('<text class="title" x="82" y="34">Convergence trace unavailable</text>')
        lines.append("</svg>")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    x_values = sorted({int(row["sample_passes"]) for row in selected})
    y_values = [float(row["minibatch_mse"]) for row in selected]
    x_min, x_max = min(x_values), max(x_values)
    y_min = min(y_values) * 0.80
    y_max = max(y_values) * 1.20
    log_min = math.log10(y_min)
    log_max = math.log10(y_max)

    def sx(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min if x_max != x_min else 1.0)) * plot_w

    def sy(value: float) -> float:
        log_value = math.log10(max(value, y_min))
        return top + (1.0 - ((log_value - log_min) / (log_max - log_min))) * plot_h

    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="82" y="34">Convergence trace, config {html.escape(config)}</text>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    decades = range(math.floor(log_min), math.ceil(log_max) + 1)
    for decade in decades:
        value = 10 ** decade
        if value < y_min or value > y_max:
            continue
        y = sy(value)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">1e{decade}</text>')
    for x_value in x_values:
        x = sx(x_value)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>')
        lines.append(f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle">{x_value}</text>')
    for variant in VARIANTS:
        series = sorted(
            [row for row in selected if row["variant"] == variant],
            key=lambda row: int(row["sample_passes"]),
        )
        if not series:
            continue
        points = [(sx(float(row["sample_passes"])), sy(float(row["minibatch_mse"]))) for row in series]
        lines.append(polyline(points, COLORS[variant]))
        for x, y in points:
            lines.append(circle(x, y, COLORS[variant]))
    lines.append(f'<text class="label" x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle">Cumulative sample-passes</text>')
    lines.append(f'<text class="label" transform="translate(24 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">minibatch MSE, log scale</text>')
    draw_legend(lines, width - right + 34, top + 8, VARIANTS)
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    summary_csv = args.summary_csv or latest_summary_csv(args.output_dir)
    output_tag = infer_output_tag(summary_csv, args.output_tag)
    summary_rows = read_csv(summary_csv)
    speedup_csv = args.output_dir / f"stm32n6_speedup_{args.date}{output_tag}.csv"
    speedup_svg = args.output_dir / f"stm32n6_speedup_{args.date}{output_tag}.svg"
    convergence_csv = args.output_dir / f"stm32n6_convergence_{args.date}{output_tag}.csv"
    convergence_svg = args.output_dir / (
        f"stm32n6_convergence_{args.date}{output_tag}_{args.convergence_config.replace('x', 'x')}.svg"
    )

    speedup_rows = write_speedup_csv(speedup_csv, summary_rows)
    write_speedup_svg(speedup_svg, speedup_rows)
    trace_rows = read_trace_rows(summary_rows)
    write_convergence_csv(convergence_csv, trace_rows)
    write_convergence_svg(convergence_svg, trace_rows, args.convergence_config)

    print(speedup_csv)
    print(speedup_svg)
    print(convergence_csv)
    print(convergence_svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
