#!/usr/bin/env python3
"""Generate dependency-free SVG plots for the EL_C_vsCpp STM32N6 sweep."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import math
import os
import re
from pathlib import Path


VARIANTS = ("legacy_c", "cpp_direct_c_backend", "cpp_m55", "cpp_generic", "rltools_generic")
LABELS = {
    "legacy_c": "C M55",
    "cpp_direct_c_backend": "C++ direct C backend",
    "cpp_m55": "C++ M55",
    "cpp_generic": "C++ Generic",
    "rltools_generic": "RLTools Generic",
}
COLORS = {
    "legacy_c": "#2f2f2f",
    "cpp_direct_c_backend": "#177245",
    "cpp_m55": "#1f62b2",
    "cpp_generic": "#b24a1f",
    "rltools_generic": "#7a3db8",
}
COMPONENTS = (
    ("zero", "zero grad", "#4f6f8f"),
    ("input_copy", "input copy", "#7c5aa6"),
    ("forward", "forward", "#2f7f5f"),
    ("loss", "loss/grad", "#d08a2f"),
    ("backward", "backward", "#b64b4b"),
    ("sample_train_step", "C train step", "#6f6f6f"),
    ("adam_update", "Adam update", "#2f64b0"),
    ("gap", "other/gap", "#c9c9c9"),
)
SHORT_LABELS = {
    "legacy_c": "C",
    "cpp_direct_c_backend": "Direct",
    "cpp_m55": "M55",
    "cpp_generic": "Generic",
    "rltools_generic": "RLTools",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    default_results_dir = repo_root / "benchmarks" / "firmware" / "stm32n6" / "el_cvscpp_ablation" / "results"
    today = dt.date.today().isoformat()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Sweep CSV produced by report_sweep_n6.py. Defaults to the latest 10-seed CSV.",
    )
    parser.add_argument("--convergence-config", default="32x32")
    parser.add_argument("--output-dir", type=Path, default=default_results_dir)
    parser.add_argument("--date", default=today)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--env-file", type=Path, default=script_dir / ".env")
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


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def to_int(value: object, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(str(value))


def active_variants_from_summary(rows: list[dict[str, str]]) -> tuple[str, ...]:
    active = [
        variant
        for variant in VARIANTS
        if any(to_int(row.get(f"{variant}_cycles_avg")) > 0 for row in rows)
    ]
    return tuple(active) if active else VARIANTS


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


def resolve_report_path(raw_path: str, project_root: Path | None) -> Path:
    if project_root is not None:
        for prefix in ("${EL_CVSCPP_PROJECT_ROOT}/", "$EL_CVSCPP_PROJECT_ROOT/"):
            if raw_path.startswith(prefix):
                return project_root / raw_path[len(prefix):]
    return Path(os.path.expandvars(raw_path))


def write_speedup_csv(path: Path, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        legacy = to_int(row.get("legacy_c_cycles_avg"))
        rltools = to_int(row.get("rltools_generic_cycles_avg"))
        baseline_variant = "legacy_c" if legacy > 0 else "rltools_generic"
        baseline = legacy if legacy > 0 else rltools
        for variant in VARIANTS:
            cycles = to_int(row.get(f"{variant}_cycles_avg"))
            if cycles <= 0:
                continue
            speedup = (baseline / cycles) if cycles else 0.0
            out.append(
                {
                    "config": row["config"],
                    "params": to_int(row.get("params")),
                    "variant": variant,
                    "baseline_variant": baseline_variant,
                    "cycles_avg": cycles,
                    "speedup_over_baseline": speedup,
                    "cycles_over_baseline": (cycles / baseline) if baseline else 0.0,
                    "speedup_over_legacy": (legacy / cycles) if legacy else 0.0,
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
                "baseline_variant",
                "cycles_avg",
                "speedup_over_baseline",
                "cycles_over_baseline",
                "speedup_over_legacy",
                "cycles_over_legacy",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(out)
    return out


def read_trace_rows(summary_rows: list[dict[str, str]],
                    project_root: Path | None) -> list[dict[str, object]]:
    traces: list[dict[str, object]] = []
    seen_logs: set[Path] = set()
    for summary in summary_rows:
        raw_paths = [summary.get("log_path", "")]
        raw_paths.extend(summary.get(f"{variant}_log_path", "") for variant in VARIANTS)
        for raw_path in raw_paths:
            if not raw_path:
                continue
            log_path = resolve_report_path(raw_path, project_root)
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
            lineterminator="\n",
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
    speeds = [float(row["speedup_over_baseline"]) for row in rows if float(row["speedup_over_baseline"]) > 0]
    if not params or not speeds:
        lines = svg_header(width, height)
        lines.append('<text class="title" x="82" y="34">Speedup trace unavailable</text>')
        lines.append("</svg>")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    x_min, x_max = min(params), max(params)
    y_min, y_max = 0.0, max(1.0, max(speeds) * 1.12)

    def sx(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min if x_max != x_min else 1.0)) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - ((value - y_min) / (y_max - y_min))) * plot_h

    baseline_variant = str(rows[0].get("baseline_variant", "legacy_c")) if rows else "legacy_c"
    baseline_label = "legacy C baseline" if baseline_variant == "legacy_c" else "RLTools Generic baseline"
    active_variants = tuple(
        variant for variant in VARIANTS if any(str(row["variant"]) == variant for row in rows)
    )

    lines = svg_header(width, height)
    lines.append(f'<text class="title" x="82" y="34">Speedup over {html.escape(baseline_label)}</text>')
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
    by_variant: dict[str, list[dict[str, object]]] = {variant: [] for variant in active_variants}
    for row in rows:
        variant = str(row["variant"])
        if variant in by_variant:
            by_variant[variant].append(row)
    for variant in active_variants:
        series = sorted(by_variant[variant], key=lambda row: int(row["params"]))
        points = [(sx(float(row["params"])), sy(float(row["speedup_over_baseline"]))) for row in series]
        lines.append(polyline(points, COLORS[variant]))
        for x, y in points:
            lines.append(circle(x, y, COLORS[variant]))
    lines.append(f'<text class="label" x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle">Network config and parameter count</text>')
    lines.append(f'<text class="label" transform="translate(22 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">baseline cycles / variant cycles</text>')
    draw_legend(lines, width - right + 34, top + 8, active_variants)
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runtime_breakdown_csv(path: Path,
                                rows: list[dict[str, str]],
                                variants: tuple[str, ...]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        config = row["config"]
        params = to_int(row.get("params"))
        for variant in variants:
            total = to_int(row.get(f"{variant}_profile_cycles_avg"))
            if total == 0:
                total = to_int(row.get(f"{variant}_cycles_avg"))
            for order, (component, label, _) in enumerate(COMPONENTS):
                cycles = to_int(row.get(f"{variant}_profile_{component}_avg"))
                out.append(
                    {
                        "config": config,
                        "params": params,
                        "variant": variant,
                        "component": component,
                        "component_label": label,
                        "component_order": order,
                        "cycles_avg": cycles,
                        "variant_cycles_avg": total,
                        "percent_of_variant": (cycles / total * 100.0) if total else 0.0,
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
                "component",
                "component_label",
                "component_order",
                "cycles_avg",
                "variant_cycles_avg",
                "percent_of_variant",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(out)
    return out


def write_runtime_breakdown_svg(path: Path,
                                rows: list[dict[str, object]],
                                variants: tuple[str, ...]) -> None:
    configs: list[str] = []
    for row in rows:
        config = str(row["config"])
        if config not in configs:
            configs.append(config)

    totals: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (str(row["config"]), str(row["variant"]))
        totals[key] = max(totals.get(key, 0.0), float(row["variant_cycles_avg"]))

    bar_w = 17
    bar_gap = 5
    group_gap = 34
    left, right, top, bottom = 90, 285, 70, 122
    group_w = len(variants) * bar_w + (len(variants) - 1) * bar_gap
    plot_w = len(configs) * group_w + (len(configs) - 1) * group_gap
    width, height = left + plot_w + right, 680
    plot_h = height - top - bottom
    y_max = max(totals.values(), default=1.0)
    y_max = max(1.0, math.ceil(y_max / 10_000_000.0) * 10_000_000.0)

    def sx(config_index: int, variant_index: int) -> float:
        return left + config_index * (group_w + group_gap) + variant_index * (bar_w + bar_gap)

    def sy(value: float) -> float:
        return top + (1.0 - value / y_max) * plot_h

    by_key_component: dict[tuple[str, str, str], float] = {}
    for row in rows:
        by_key_component[(str(row["config"]), str(row["variant"]), str(row["component"]))] = float(row["cycles_avg"])

    lines = svg_header(width, height)
    lines.append('<text class="title" x="90" y="34">Training-loop component breakdown</text>')
    lines.append(
        '<text class="label" x="90" y="54">Average DWT cycles per measured run; setup, warm-up, trace, export, and serial I/O excluded.</text>'
    )
    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    for i in range(6):
        value = y_max * i / 5
        y = sy(value)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value / 1_000_000:.0f}M</text>')
    for ci, config in enumerate(configs):
        group_x = left + ci * (group_w + group_gap)
        group_center = group_x + group_w / 2
        lines.append(
            f'<text class="label" x="{group_center:.1f}" y="{top + plot_h + 76}" '
            f'text-anchor="middle">{html.escape(config)}</text>'
        )
        for vi, variant in enumerate(variants):
            x = sx(ci, vi)
            accum = 0.0
            for component, label, color in COMPONENTS:
                cycles = by_key_component.get((config, variant, component), 0.0)
                if cycles <= 0:
                    continue
                y_top = sy(accum + cycles)
                y_bottom = sy(accum)
                segment_h = max(0.6, y_bottom - y_top)
                title = (
                    f"{config} {LABELS[variant]} {label}: {cycles:,.0f} cycles; "
                    f"total: {totals.get((config, variant), 0.0):,.0f}"
                )
                lines.append(
                    f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bar_w}" '
                    f'height="{segment_h:.1f}" fill="{color}"><title>{html.escape(title)}</title></rect>'
                )
                accum += cycles
            total = totals.get((config, variant), 0.0)
            if total > 0:
                lines.append(
                    f'<rect x="{x:.1f}" y="{sy(total):.1f}" width="{bar_w}" '
                    f'height="{sy(0.0) - sy(total):.1f}" fill="none" stroke="#222" stroke-width=".45" opacity=".55"/>'
                )
            label_x = x + bar_w / 2
            label_y = top + plot_h + 22
            lines.append(
                f'<text class="tick" x="{label_x:.1f}" y="{label_y}" text-anchor="end" '
                f'transform="rotate(-45 {label_x:.1f} {label_y})">{html.escape(SHORT_LABELS[variant])}</text>'
            )
    lines.append(
        f'<text class="label" transform="translate(24 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">cycles per measured run</text>'
    )
    legend_x = left + plot_w + 44
    legend_y = top + 8
    lines.append(f'<text class="legend" x="{legend_x}" y="{legend_y - 16}">Components</text>')
    for i, (_, label, color) in enumerate(COMPONENTS):
        y = legend_y + i * 21
        lines.append(f'<rect x="{legend_x}" y="{y - 12}" width="16" height="16" fill="{color}"/>')
        lines.append(f'<text class="legend" x="{legend_x + 24}" y="{y + 1}">{html.escape(label)}</text>')
    variant_y = legend_y + len(COMPONENTS) * 21 + 42
    lines.append(f'<text class="legend" x="{legend_x}" y="{variant_y - 16}">Variant order</text>')
    for i, variant in enumerate(variants):
        y = variant_y + i * 19
        lines.append(f'<text class="legend" x="{legend_x}" y="{y}">{html.escape(SHORT_LABELS[variant])}: {html.escape(LABELS[variant])}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_markdown_plots(summary_csv: Path,
                          speedup_svg: Path,
                          convergence_svg: Path,
                          runtime_breakdown_svg: Path,
                          elf_svg: Path | None = None) -> None:
    md_path = summary_csv.with_suffix(".md")
    if not md_path.exists():
        return
    start = "<!-- plots:start -->"
    end = "<!-- plots:end -->"
    text = md_path.read_text(encoding="utf-8")
    elf_line = ""
    if elf_svg is not None and elf_svg.exists():
        elf_line = f"![Firmware ELF component breakdown]({elf_svg.name})"
    elif start in text and end in text:
        old_section = text.split(start, 1)[1].split(end, 1)[0]
        for line in old_section.splitlines():
            if line.startswith("![Firmware ELF component breakdown]"):
                elf_line = line
                break

    section_lines = [
        start,
        "## Generated plots",
        "",
        f"![Speedup curve]({speedup_svg.name})",
        "",
        f"![Training-loop component breakdown]({runtime_breakdown_svg.name})",
        "",
        f"![Convergence trace]({convergence_svg.name})",
    ]
    if elf_line:
        section_lines.extend(["", elf_line])
    section_lines.extend([end, ""])
    section = "\n".join(section_lines)
    if start in text and end in text:
        before = text.split(start, 1)[0].rstrip()
        after = text.split(end, 1)[1].lstrip()
        text = before + "\n\n" + section + ("\n" + after if after else "")
    else:
        text = text.rstrip() + "\n\n" + section
    md_path.write_text(text, encoding="utf-8")


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
    active_variants = tuple(
        variant for variant in VARIANTS if any(row["variant"] == variant for row in selected)
    )
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
    draw_legend(lines, width - right + 34, top + 8, active_variants)
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    env = load_env(args.env_file)
    raw_project_root = os.environ.get("EL_CVSCPP_PROJECT_ROOT", env.get("EL_CVSCPP_PROJECT_ROOT", ""))
    project_root = Path(raw_project_root).expanduser().resolve() if raw_project_root else None
    summary_csv = args.summary_csv or latest_summary_csv(args.output_dir)
    output_tag = infer_output_tag(summary_csv, args.output_tag)
    summary_rows = read_csv(summary_csv)
    speedup_csv = args.output_dir / f"stm32n6_speedup_{args.date}{output_tag}.csv"
    speedup_svg = args.output_dir / f"stm32n6_speedup_{args.date}{output_tag}.svg"
    convergence_csv = args.output_dir / f"stm32n6_convergence_{args.date}{output_tag}.csv"
    convergence_svg = args.output_dir / (
        f"stm32n6_convergence_{args.date}{output_tag}_{args.convergence_config.replace('x', 'x')}.svg"
    )
    runtime_breakdown_csv = args.output_dir / f"stm32n6_training_component_breakdown_{args.date}{output_tag}.csv"
    runtime_breakdown_svg = args.output_dir / f"stm32n6_training_component_breakdown_{args.date}{output_tag}.svg"
    elf_svg = args.output_dir / f"stm32n6_elf_component_breakdown_{args.date}{output_tag}.svg"
    active_variants = active_variants_from_summary(summary_rows)

    speedup_rows = write_speedup_csv(speedup_csv, summary_rows)
    write_speedup_svg(speedup_svg, speedup_rows)
    runtime_breakdown_rows = write_runtime_breakdown_csv(runtime_breakdown_csv, summary_rows, active_variants)
    write_runtime_breakdown_svg(runtime_breakdown_svg, runtime_breakdown_rows, active_variants)
    trace_rows = read_trace_rows(summary_rows, project_root)
    write_convergence_csv(convergence_csv, trace_rows)
    write_convergence_svg(convergence_svg, trace_rows, args.convergence_config)
    update_markdown_plots(summary_csv, speedup_svg, convergence_svg, runtime_breakdown_svg, elf_svg)

    print(speedup_csv)
    print(speedup_svg)
    print(runtime_breakdown_csv)
    print(runtime_breakdown_svg)
    print(convergence_csv)
    print(convergence_svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
