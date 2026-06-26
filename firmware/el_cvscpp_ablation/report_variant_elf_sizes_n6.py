#!/usr/bin/env python3
"""Report per-variant STM32N6 ELF sizes for the EL_C_vsCpp ablation."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path


DEFAULT_CONFIGS = ("8x8", "16x8", "16x16", "32x16", "32x32", "64x32")
VARIANTS = ("legacy_c", "cpp_direct_c_backend", "cpp_m55", "cpp_generic", "rltools_generic")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=script_dir / ".env")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--build-config")
    parser.add_argument("--appli-project-name")
    parser.add_argument("--input-features", type=int)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument(
        "--combined-csv",
        type=Path,
        help="Optional combined sweep CSV used as a fallback for model-size fields.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help="Output path without extension. .csv and .md are written.",
    )
    return parser.parse_args()


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


def to_int(value: object | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def find_one(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[0] if matches else None


def parse_size_file(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {"elf_text": 0, "elf_data": 0, "elf_bss": 0, "elf_dec": 0}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 4 and all(part.isdigit() for part in parts[:4]):
            return {
                "elf_text": int(parts[0]),
                "elf_data": int(parts[1]),
                "elf_bss": int(parts[2]),
                "elf_dec": int(parts[3]),
            }
    return {"elf_text": 0, "elf_data": 0, "elf_bss": 0, "elf_dec": 0}


def parse_begin(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        kind, values = parse_kv_line(line)
        if kind == "BEGIN":
            return values
    return {}


def parse_variant_log(path: Path, variant: str) -> dict[str, int]:
    summary: dict[str, str] = {}
    done: dict[str, str] = {}
    if not path.exists():
        return {
            "seeds": 0,
            "cycles_avg": 0,
            "cycles_min": 0,
            "cycles_max": 0,
            "runtime_status": -999,
            "runtime_done_status": -999,
        }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        kind, values = parse_kv_line(line)
        if values.get("variant") != variant:
            continue
        if kind == "SUMMARY":
            summary = values
        elif kind == "DONE":
            done = values
    return {
        "seeds": to_int(done.get("seeds"), to_int(summary.get("seeds"))),
        "cycles_avg": to_int(summary.get("cycles_avg")),
        "cycles_min": to_int(summary.get("cycles_min")),
        "cycles_max": to_int(summary.get("cycles_max")),
        "runtime_status": to_int(summary.get("status"), -999),
        "runtime_done_status": to_int(done.get("status"), -999),
    }


def latest_combined_csv(script_dir: Path) -> Path | None:
    matches = sorted(
        (script_dir / "results").glob("stm32n6_sweep_*_10seed.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    return matches[-1] if matches else None


def read_combined_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["config"]: row for row in csv.DictReader(f) if row.get("config")}


def value(begin: dict[str, str],
          combined: dict[str, str],
          begin_key: str,
          combined_key: str) -> int:
    return to_int(begin.get(begin_key), to_int(combined.get(combined_key)))


def model_size_values(variant: str,
                      begin: dict[str, str],
                      combined: dict[str, str]) -> dict[str, object]:
    if variant == "legacy_c":
        arena = value(begin, combined, "legacy_c_arena", "legacy_c_arena_bytes")
        control = value(begin, combined, "legacy_c_control", "legacy_c_control_bytes")
        return {
            "model_size_kind": "arena+control",
            "model_state_bytes": arena + control,
            "arena_or_required_memory": arena,
            "object_or_control_bytes": control,
        }
    if variant == "cpp_direct_c_backend":
        required = value(begin, combined, "cpp_direct_required_memory", "cpp_direct_required_memory")
        obj = value(begin, combined, "cpp_direct_model_object", "cpp_direct_model_object")
        return {
            "model_size_kind": "required_memory/model_object",
            "model_state_bytes": obj,
            "arena_or_required_memory": required,
            "object_or_control_bytes": obj,
        }
    if variant == "cpp_m55":
        required = value(begin, combined, "cpp_m55_required_memory", "cpp_m55_required_memory")
        obj = value(begin, combined, "cpp_m55_model_object", "cpp_m55_model_object")
        return {
            "model_size_kind": "required_memory/model_object",
            "model_state_bytes": obj,
            "arena_or_required_memory": required,
            "object_or_control_bytes": obj,
        }
    if variant == "cpp_generic":
        required = value(begin, combined, "cpp_generic_required_memory", "cpp_generic_required_memory")
        obj = value(begin, combined, "cpp_generic_model_object", "cpp_generic_model_object")
        return {
            "model_size_kind": "required_memory/model_object",
            "model_state_bytes": obj,
            "arena_or_required_memory": required,
            "object_or_control_bytes": obj,
        }
    if variant == "rltools_generic":
        required = value(begin, combined, "rltools_static_state", "rltools_static_state")
        obj = value(begin, combined, "rltools_model_object", "rltools_model_object")
        return {
            "model_size_kind": "static_state/model_object",
            "model_state_bytes": obj,
            "arena_or_required_memory": required,
            "object_or_control_bytes": obj,
        }
    raise ValueError(f"unknown variant: {variant}")


def combined_runtime_values(variant: str, combined: dict[str, str]) -> dict[str, int]:
    if not combined:
        return {}
    prefix = "legacy_c" if variant == "legacy_c" else variant
    avg = to_int(combined.get(f"{prefix}_cycles_avg"))
    if avg == 0:
        return {}
    return {
        "seeds": to_int(combined.get("seeds")),
        "cycles_avg": avg,
        "cycles_min": to_int(combined.get(f"{prefix}_cycles_min")),
        "cycles_max": to_int(combined.get(f"{prefix}_cycles_max")),
        "runtime_status": to_int(combined.get("status"), -999),
        "runtime_done_status": to_int(combined.get("status"), -999),
    }


def build_row(project_root: Path,
              build_config: str,
              appli_project_name: str,
              input_features: int,
              config: str,
              variant: str,
              combined_row: dict[str, str]) -> dict[str, object]:
    h1, h2 = config.split("x", 1)
    config_suffix = f"{input_features}_{h1}x{h2}_1"
    combined_build_dir = project_root / "STM32CubeIDE" / "Appli" / build_config / config_suffix
    build_dir = combined_build_dir / variant
    size_path = find_one(build_dir, "*.size.txt")
    elf_path = build_dir / f"{appli_project_name}_{config_suffix}_{variant}.elf"
    if not elf_path.exists():
        elf_path = find_one(build_dir, "*.elf") or elf_path
    begin = parse_begin(combined_build_dir / "serial.log")
    if not begin:
        begin = parse_begin(build_dir / "serial.log")

    row: dict[str, object] = {
        "config": config,
        "input_features": to_int(begin.get("input_features"), input_features),
        "variant": variant,
        "warmups": to_int(begin.get("warmups"), to_int(combined_row.get("warmups"))),
        "build_dir": str(build_dir),
        "serial_path": str(build_dir / "serial.log"),
        "size_path": str(size_path or ""),
        "elf_path": str(elf_path),
        "elf_file_bytes": elf_path.stat().st_size if elf_path.exists() else 0,
    }
    runtime = parse_variant_log(build_dir / "serial.log", variant)
    runtime_source = "per_variant_log"
    if int(runtime.get("cycles_avg", 0)) == 0:
        fallback = combined_runtime_values(variant, combined_row)
        if fallback:
            runtime = fallback
            runtime_source = "combined_all_log"
        else:
            runtime_source = "missing"
    row.update(runtime)
    row["runtime_source"] = runtime_source
    row.update(model_size_values(variant, begin, combined_row))
    row.update(parse_size_file(size_path))
    return row


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def add_ratios(rows: list[dict[str, object]]) -> None:
    legacy_by_config = {
        str(row["config"]): row for row in rows if row["variant"] == "legacy_c"
    }
    for row in rows:
        legacy = legacy_by_config.get(str(row["config"]), {})
        row["elf_text_over_legacy"] = ratio(
            int(row["elf_text"]), int(legacy.get("elf_text", 0))
        )
        row["elf_dec_over_legacy"] = ratio(
            int(row["elf_dec"]), int(legacy.get("elf_dec", 0))
        )
        row["elf_file_over_legacy"] = ratio(
            int(row["elf_file_bytes"]), int(legacy.get("elf_file_bytes", 0))
        )
        row["cycles_over_legacy"] = ratio(
            int(row["cycles_avg"]), int(legacy.get("cycles_avg", 0))
        )


def fmt_int(value: object) -> str:
    return f"{int(value):,}"


def fmt_ratio(value: object) -> str:
    return f"{float(value):.3f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
        "input_features",
        "variant",
        "model_size_kind",
        "seeds",
        "warmups",
        "cycles_avg",
        "cycles_min",
        "cycles_max",
        "cycles_over_legacy",
        "runtime_source",
        "runtime_status",
        "runtime_done_status",
        "model_state_bytes",
        "arena_or_required_memory",
        "object_or_control_bytes",
        "elf_text",
        "elf_data",
        "elf_bss",
        "elf_dec",
        "elf_file_bytes",
        "elf_text_over_legacy",
        "elf_dec_over_legacy",
        "elf_file_over_legacy",
        "build_dir",
        "serial_path",
        "size_path",
        "elf_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_for(rows: list[dict[str, object]], config: str, variant: str) -> dict[str, object]:
    for row in rows:
        if row["config"] == config and row["variant"] == variant:
            return row
    return {}


def cell(row: dict[str, object]) -> str:
    if not row:
        return "missing"
    return (
        f"{fmt_int(row['elf_dec'])}/{fmt_int(row['elf_file_bytes'])} "
        f"({fmt_ratio(row['elf_dec_over_legacy'])}x)"
    )


def runtime_cell(row: dict[str, object]) -> str:
    if not row or int(row.get("cycles_avg", 0)) == 0:
        return "missing"
    return f"{fmt_int(row['cycles_avg'])} ({fmt_ratio(row['cycles_over_legacy'])}x)"


def write_markdown(path: Path,
                   rows: list[dict[str, object]],
                   configs: list[str],
                   combined_csv: Path | None,
                   input_features: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_done = all(
        int(row.get("runtime_status", -999)) == 0
        and int(row.get("runtime_done_status", -999)) == 0
        for row in rows
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# STM32N6 EL_C_vsCpp Per-Variant Sweep - {dt.date.today().isoformat()}\n\n")
        f.write("Build: one firmware ELF per variant and per network size, all with `-Ofast`.\n")
        f.write(f"Network input features: `{input_features}`.\n")
        f.write("Runtime: Adam, 10 seeds, 2 warm-up runs per seed/variant, batch 256, 1024 rollout samples, 2 epochs, 8 optimizer steps.\n")
        f.write("Timing excludes setup, warm-up, convergence tracing, export, and numerical comparisons.\n")
        f.write("Runtime columns use per-variant logs when available, otherwise the combined `variant=all` sweep.\n")
        f.write("ELF columns report `arm-none-eabi-size dec` and on-disk ELF bytes.\n")
        f.write("Each ELF still includes the common benchmark harness and static rollout buffers.\n")
        f.write(f"All runtime sources completed with `status=0`: `{int(all_done)}`.\n")
        if combined_csv is not None:
            f.write(f"Model-size metadata fallback: `{combined_csv}`.\n")
        f.write("\n")
        f.write("| Config | Input | C cycles | Direct C-backend cycles | Direct/C | C++ M55 cycles | M55/C | C++ Generic cycles | Generic/C | RLTools Generic cycles | RLTools/C |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for config in configs:
            legacy = row_for(rows, config, "legacy_c")
            direct = row_for(rows, config, "cpp_direct_c_backend")
            m55 = row_for(rows, config, "cpp_m55")
            generic = row_for(rows, config, "cpp_generic")
            rltools = row_for(rows, config, "rltools_generic")
            f.write(
                f"| {config} | {input_features} | {fmt_int(legacy.get('cycles_avg', 0))} | "
                f"{fmt_int(direct.get('cycles_avg', 0))} | "
                f"{fmt_ratio(direct.get('cycles_over_legacy', 0.0))} | "
                f"{fmt_int(m55.get('cycles_avg', 0))} | "
                f"{fmt_ratio(m55.get('cycles_over_legacy', 0.0))} | "
                f"{fmt_int(generic.get('cycles_avg', 0))} | "
                f"{fmt_ratio(generic.get('cycles_over_legacy', 0.0))} | "
                f"{fmt_int(rltools.get('cycles_avg', 0))} | "
                f"{fmt_ratio(rltools.get('cycles_over_legacy', 0.0))} |\n"
            )
        f.write("\n")
        f.write(
            "| Config | Input | C model state | Direct model state | M55 model state | "
            "Generic model state | RLTools model state | C ELF dec/file | Direct ELF dec/file | "
            "M55 ELF dec/file | Generic ELF dec/file | RLTools ELF dec/file |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for config in configs:
            legacy = row_for(rows, config, "legacy_c")
            direct = row_for(rows, config, "cpp_direct_c_backend")
            m55 = row_for(rows, config, "cpp_m55")
            generic = row_for(rows, config, "cpp_generic")
            rltools = row_for(rows, config, "rltools_generic")
            f.write(
                f"| {config} | {input_features} | {fmt_int(legacy.get('model_state_bytes', 0))} | "
                f"{fmt_int(direct.get('model_state_bytes', 0))} | "
                f"{fmt_int(m55.get('model_state_bytes', 0))} | "
                f"{fmt_int(generic.get('model_state_bytes', 0))} | "
                f"{fmt_int(rltools.get('model_state_bytes', 0))} | "
                f"{cell(legacy)} | {cell(direct)} | {cell(m55)} | {cell(generic)} | "
                f"{cell(rltools)} |\n"
            )
        f.write("\nRaw per-variant rows, `.size.txt` paths, and ELF paths are in the CSV.\n")


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    env = load_env(args.env_file)
    project_root = args.project_root or Path(
        os.environ.get("EL_CVSCPP_PROJECT_ROOT", env.get("EL_CVSCPP_PROJECT_ROOT", ""))
    )
    build_config = args.build_config or os.environ.get(
        "EL_CVSCPP_BUILD_CONFIG", env.get("EL_CVSCPP_BUILD_CONFIG", "EL_C_vsCpp_Ablation_DEBUG")
    )
    appli_project_name = args.appli_project_name or os.environ.get(
        "EL_CVSCPP_APPLI_PROJECT_NAME", env.get("EL_CVSCPP_APPLI_PROJECT_NAME", "EL_C_vsCpp_Appli")
    )
    input_features = args.input_features or int(
        os.environ.get("EL_CVSCPP_INPUT_FEATURES", env.get("EL_CVSCPP_INPUT_FEATURES", "3"))
    )
    if not str(project_root):
        raise SystemExit("project root missing; pass --project-root or set EL_CVSCPP_PROJECT_ROOT")

    output_stem = args.output_stem or (
        script_dir
        / "results"
        / f"stm32n6_variant_elf_sizes_{dt.date.today().isoformat()}_input{input_features}"
    )

    combined_csv = args.combined_csv or latest_combined_csv(script_dir)
    combined_rows = read_combined_rows(combined_csv)

    rows: list[dict[str, object]] = []
    for config in args.configs:
        for variant in args.variants:
            rows.append(
                build_row(
                    project_root,
                    build_config,
                    appli_project_name,
                    input_features,
                    config,
                    variant,
                    combined_rows.get(config, {}),
                )
            )
    add_ratios(rows)
    write_csv(output_stem.with_suffix(".csv"), rows)
    write_markdown(output_stem.with_suffix(".md"), rows, list(args.configs), combined_csv, input_features)
    print(output_stem.with_suffix(".csv"))
    print(output_stem.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
