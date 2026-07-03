#!/usr/bin/env python3
"""Generate STM32N6 firmware ELF component-breakdown SVG plots."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import math
import re
from pathlib import Path


VARIANTS = ("legacy_c", "cpp_direct_c_backend", "cpp_m55", "cpp_generic", "rltools_generic")
PLOT_VARIANTS = tuple(variant for variant in VARIANTS if variant != "cpp_direct_c_backend")
VARIANT_LABELS = {
    "legacy_c": "C M55",
    "cpp_direct_c_backend": "C++ direct C backend",
    "cpp_m55": "C++ M55",
    "cpp_generic": "C++ Generic",
    "rltools_generic": "RLTools Generic",
}
VARIANT_SHORT = {
    "legacy_c": "C",
    "cpp_direct_c_backend": "Direct",
    "cpp_m55": "M55",
    "cpp_generic": "Generic",
    "rltools_generic": "RLTools",
}
COMPONENTS = (
    ("elf_text", "text", "#356bb0"),
    ("elf_data", "data", "#e0a22e"),
    ("elf_bss", "bss", "#4f9f68"),
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    default_results_dir = repo_root / "benchmarks" / "firmware" / "stm32n6" / "el_cvscpp_ablation" / "results"
    today = dt.date.today().isoformat()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="CSV produced by report_sweep_n6.py. Defaults to the latest 10-seed sweep CSV.",
    )
    parser.add_argument("--size-csv", type=Path, dest="summary_csv", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=default_results_dir)
    parser.add_argument("--date", default=today)
    parser.add_argument("--output-tag", default=None)
    return parser.parse_args()


def latest_summary_csv(results_dir: Path) -> Path:
    matches = sorted(
        results_dir.glob("stm32n6_sweep_*_10seed.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise SystemExit("No sweep CSV found; run report_sweep_n6.py first.")
    return matches[-1]


def infer_output_tag(summary_csv: Path, explicit_tag: str | None) -> str:
    if explicit_tag is not None:
        return explicit_tag
    match = re.search(r"_input(\d+)(?:_|$)", summary_csv.stem)
    return f"_input{match.group(1)}" if match else ""


def to_int(value: object, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(str(value))


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for summary in csv.DictReader(f):
            for variant in PLOT_VARIANTS:
                row = {
                    "config": summary.get("config", ""),
                    "input_features": to_int(summary.get("input_features")),
                    "variant": variant,
                    "model_size_kind": summary.get(f"{variant}_model_size_kind", ""),
                    "model_state_bytes": to_int(summary.get(f"{variant}_model_state_bytes")),
                    "cycles_avg": to_int(summary.get(f"{variant}_cycles_avg")),
                    "elf_text": to_int(summary.get(f"{variant}_elf_text")),
                    "elf_data": to_int(summary.get(f"{variant}_elf_data")),
                    "elf_bss": to_int(summary.get(f"{variant}_elf_bss")),
                    "elf_dec": to_int(summary.get(f"{variant}_elf_dec")),
                    "elf_file_bytes": to_int(summary.get(f"{variant}_elf_file_bytes")),
                }
                if to_int(row["elf_dec"]) > 0 or to_int(row["cycles_avg"]) > 0:
                    rows.append(row)
    return rows


def write_breakdown_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "input_features",
                "variant",
                "model_size_kind",
                "model_state_bytes",
                "cycles_avg",
                "elf_text",
                "elf_data",
                "elf_bss",
                "elf_dec",
                "elf_file_bytes",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def fmt_kib(value: float) -> str:
    return f"{value / 1024.0:.0f} KiB"


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            '<style>'
            'text{font-family:Arial,Helvetica,sans-serif;fill:#202020}'
            '.axis{stroke:#222;stroke-width:1.2}'
            '.grid{stroke:#dedede;stroke-width:1}'
            '.title{font-size:20px;font-weight:700}'
            '.subtitle{font-size:12px;fill:#555}'
            '.label{font-size:13px}'
            '.tick{font-size:11px}'
            '.legend{font-size:12px}'
            '.baroutline{fill:none;stroke:#222;stroke-width:.5;opacity:.45}'
            '</style>'
        ),
    ]


def row_for(rows: list[dict[str, object]], config: str, variant: str) -> dict[str, object] | None:
    for row in rows:
        if row["config"] == config and row["variant"] == variant:
            return row
    return None


def write_svg(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        lines = svg_header(1100, 420)
        lines.append('<text class="title" x="82" y="34">STM32N6 firmware ELF component breakdown unavailable</text>')
        lines.append("</svg>")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    configs: list[str] = []
    for row in rows:
        config = str(row["config"])
        if config not in configs:
            configs.append(config)
    variants = tuple(
        variant for variant in PLOT_VARIANTS if any(row["variant"] == variant for row in rows)
    )

    bar_w = 16
    bar_gap = 5
    group_gap = 34
    left = 82
    right = 275
    top = 66
    bottom = 112
    group_w = len(variants) * bar_w + (len(variants) - 1) * bar_gap
    plot_w = len(configs) * group_w + (len(configs) - 1) * group_gap
    width = left + plot_w + right
    height = 650
    plot_h = height - top - bottom
    max_value = max((int(row["elf_dec"]) for row in rows), default=1)
    y_max = math.ceil(max_value / 16384.0) * 16384.0
    y_max = max(y_max, 16384.0)

    def sx(config_index: int, variant_index: int) -> float:
        return left + config_index * (group_w + group_gap) + variant_index * (bar_w + bar_gap)

    def sy(value: float) -> float:
        return top + (1.0 - value / y_max) * plot_h

    lines = svg_header(width, height)
    lines.append('<text class="title" x="82" y="34">STM32N6 firmware ELF component breakdown</text>')
    lines.append(
        '<text class="subtitle" x="82" y="52">'
        'One ELF per variant and topology; stacked bars show arm-none-eabi-size text/data/bss.'
        '</text>'
    )
    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')

    for i in range(6):
        value = y_max * i / 5
        y = sy(value)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        lines.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{fmt_kib(value)}</text>')

    for ci, config in enumerate(configs):
        group_x = left + ci * (group_w + group_gap)
        group_center = group_x + group_w / 2
        lines.append(
            f'<text class="label" x="{group_center:.1f}" y="{top + plot_h + 72}" '
            f'text-anchor="middle">{html.escape(config)}</text>'
        )
        for vi, variant in enumerate(variants):
            row = row_for(rows, config, variant)
            if row is None:
                continue
            x = sx(ci, vi)
            accum = 0.0
            for key, label, color in COMPONENTS:
                value = float(row[key])
                y_top = sy(accum + value)
                y_bottom = sy(accum)
                height_segment = max(0.5, y_bottom - y_top)
                title = (
                    f"{config} {VARIANT_LABELS[variant]} {label}: {int(value):,} bytes; "
                    f"ELF dec: {int(row['elf_dec']):,}; model state: {int(row['model_state_bytes']):,}"
                )
                lines.append(
                    f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bar_w}" '
                    f'height="{height_segment:.1f}" fill="{color}"><title>{html.escape(title)}</title></rect>'
                )
                accum += value
            lines.append(
                f'<rect class="baroutline" x="{x:.1f}" y="{sy(float(row["elf_dec"])):.1f}" '
                f'width="{bar_w}" height="{sy(0) - sy(float(row["elf_dec"])):.1f}"/>'
            )
            label_x = x + bar_w / 2
            lines.append(
                f'<text class="tick" x="{label_x:.1f}" y="{top + plot_h + 20}" '
                f'text-anchor="end" transform="rotate(-45 {label_x:.1f} {top + plot_h + 20})">'
                f'{html.escape(VARIANT_SHORT[variant])}</text>'
            )

    lines.append(
        f'<text class="label" transform="translate(24 {top + plot_h / 2:.1f}) rotate(-90)" '
        'text-anchor="middle">ELF size components</text>'
    )
    legend_x = left + plot_w + 44
    legend_y = top + 8
    lines.append(f'<text class="legend" x="{legend_x}" y="{legend_y - 16}">ELF components</text>')
    for i, (_, label, color) in enumerate(COMPONENTS):
        y = legend_y + i * 22
        lines.append(f'<rect x="{legend_x}" y="{y - 12}" width="16" height="16" fill="{color}"/>')
        lines.append(f'<text class="legend" x="{legend_x + 24}" y="{y + 1}">{label}</text>')
    variant_y = legend_y + 92
    lines.append(f'<text class="legend" x="{legend_x}" y="{variant_y - 16}">Variant order</text>')
    for i, variant in enumerate(variants):
        y = variant_y + i * 19
        lines.append(f'<text class="legend" x="{legend_x}" y="{y}">{html.escape(VARIANT_SHORT[variant])}: {html.escape(VARIANT_LABELS[variant])}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_markdown_plots(summary_csv: Path, elf_svg: Path) -> None:
    md_path = summary_csv.with_suffix(".md")
    if not md_path.exists():
        return
    start = "<!-- plots:start -->"
    end = "<!-- plots:end -->"
    image_line = f"![Firmware ELF component breakdown]({elf_svg.name})"
    text = md_path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        section = "\n".join([start, "## Generated plots", "", image_line, end, ""])
        md_path.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")
        return

    before, rest = text.split(start, 1)
    section, after = rest.split(end, 1)
    lines = section.strip().splitlines()
    filtered = [
        line for line in lines
        if not line.startswith("![Firmware ELF component breakdown]")
    ]
    if filtered and filtered[-1] != "":
        filtered.append("")
    filtered.append(image_line)
    new_section = "\n".join([start, *filtered, end])
    md_path.write_text(before.rstrip() + "\n\n" + new_section + after, encoding="utf-8")


def main() -> int:
    args = parse_args()
    summary_csv = args.summary_csv or latest_summary_csv(args.output_dir)
    output_tag = infer_output_tag(summary_csv, args.output_tag)
    rows = read_rows(summary_csv)
    out_csv = args.output_dir / f"stm32n6_elf_component_breakdown_{args.date}{output_tag}.csv"
    out_svg = args.output_dir / f"stm32n6_elf_component_breakdown_{args.date}{output_tag}.svg"
    write_breakdown_csv(out_csv, rows)
    write_svg(out_svg, rows)
    update_markdown_plots(summary_csv, out_svg)
    print(out_csv)
    print(out_svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
