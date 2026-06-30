#!/usr/bin/env python3
"""Build CSV and Markdown reports from STM32N6 per-variant sweep artifacts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path


DEFAULT_CONFIGS = ("8x8", "16x8", "16x16", "32x16", "32x32", "64x32")
VARIANTS = ("legacy_c", "cpp_direct_c_backend", "cpp_m55", "cpp_generic", "rltools_generic")
PUBLIC_VARIANTS = ("cpp_m55", "cpp_generic", "rltools_generic")
VARIANT_LABELS = {
    "legacy_c": "C M55",
    "cpp_direct_c_backend": "Direct C-backend",
    "cpp_m55": "C++ M55",
    "cpp_generic": "C++ Generic",
    "rltools_generic": "RLTools Generic Batch",
}
PROFILE_COMPONENTS = (
    "zero",
    "input_copy",
    "forward",
    "loss",
    "backward",
    "sample_train_step",
    "adam_update",
    "component_sum",
    "gap",
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=script_dir / ".env")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--build-config")
    parser.add_argument("--appli-project-name")
    parser.add_argument("--input-features", type=int)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variants to include. Defaults to EL_CVSCPP_SWEEP_VARIANTS or the public C++/RLTools set.",
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


def resolve_variants(args: argparse.Namespace, env: dict[str, str]) -> tuple[str, ...]:
    raw_variants = args.variants
    if raw_variants is None:
        env_value = os.environ.get("EL_CVSCPP_SWEEP_VARIANTS", env.get("EL_CVSCPP_SWEEP_VARIANTS", ""))
        raw_variants = env_value.split() if env_value else list(PUBLIC_VARIANTS)
    variants = tuple(raw_variants)
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s): {' '.join(unknown)}")
    return variants


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


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def find_one(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    return matches[0] if matches else None


def portable_project_path(path: Path | None, project_root: Path) -> str:
    if path is None:
        return ""
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return path.as_posix()
    return "${EL_CVSCPP_PROJECT_ROOT}/" + relative.as_posix()


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


def parse_serial_log(path: Path) -> dict[str, object]:
    begin: dict[str, str] = {}
    done: dict[str, str] = {}
    summaries: dict[str, dict[str, str]] = {}
    result_seeds: dict[str, set[int]] = {variant: set() for variant in VARIANTS}

    if not path.exists():
        return {
            "begin": begin,
            "done": done,
            "summaries": summaries,
            "result_seeds": result_seeds,
            "log_exists": False,
        }

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        kind, values = parse_kv_line(line)
        if kind == "BEGIN":
            begin = values
        elif kind == "DONE":
            done = values
        elif kind == "SUMMARY":
            variant = values.get("variant")
            if variant:
                summaries[variant] = values
        elif kind == "RESULT":
            variant = values.get("variant")
            seed = values.get("seed")
            if variant in result_seeds and seed is not None:
                result_seeds[variant].add(int(seed))

    return {
        "begin": begin,
        "done": done,
        "summaries": summaries,
        "result_seeds": result_seeds,
        "log_exists": True,
    }


def profile_values(prefix: str, summary: dict[str, str]) -> dict[str, int]:
    values = {
        f"{prefix}_profile_{component}_avg": to_int(summary.get(f"prof_{component}_avg"))
        for component in PROFILE_COMPONENTS
    }
    values[f"{prefix}_profile_cycles_avg"] = to_int(summary.get("prof_cycles_avg"))
    return values


def config_suffix(input_features: int, config: str) -> str:
    h1, h2 = config.split("x", 1)
    return f"{input_features}_{h1}x{h2}_1"


def variant_build_dir(project_root: Path,
                      build_config: str,
                      input_features: int,
                      config: str,
                      variant: str) -> Path:
    return (
        project_root
        / "STM32CubeIDE"
        / "Appli"
        / build_config
        / config_suffix(input_features, config)
        / variant
    )


def model_size_values(variant: str,
                      begin: dict[str, str],
                      summary: dict[str, str]) -> dict[str, object]:
    if variant == "legacy_c":
        arena = to_int(begin.get("legacy_c_arena"), to_int(summary.get("arena_bytes")))
        control = to_int(begin.get("legacy_c_control"), to_int(summary.get("object_bytes")))
        return {
            "model_size_kind": "arena+control",
            "model_state_bytes": arena + control,
            "arena_or_required_memory": arena,
            "object_or_control_bytes": control,
        }
    if variant == "cpp_direct_c_backend":
        required = to_int(
            begin.get("cpp_direct_required_memory"), to_int(summary.get("arena_bytes"))
        )
        obj = to_int(begin.get("cpp_direct_model_object"), to_int(summary.get("object_bytes")))
        return {
            "model_size_kind": "required_memory/model_object",
            "model_state_bytes": obj,
            "arena_or_required_memory": required,
            "object_or_control_bytes": obj,
        }
    if variant == "cpp_m55":
        required = to_int(begin.get("cpp_m55_required_memory"), to_int(summary.get("arena_bytes")))
        obj = to_int(begin.get("cpp_m55_model_object"), to_int(summary.get("object_bytes")))
        return {
            "model_size_kind": "required_memory/model_object",
            "model_state_bytes": obj,
            "arena_or_required_memory": required,
            "object_or_control_bytes": obj,
        }
    if variant == "cpp_generic":
        required = to_int(
            begin.get("cpp_generic_required_memory"), to_int(summary.get("arena_bytes"))
        )
        obj = to_int(begin.get("cpp_generic_model_object"), to_int(summary.get("object_bytes")))
        return {
            "model_size_kind": "required_memory/model_object",
            "model_state_bytes": obj,
            "arena_or_required_memory": required,
            "object_or_control_bytes": obj,
        }
    if variant == "rltools_generic":
        state = to_int(begin.get("rltools_static_state"), to_int(summary.get("arena_bytes")))
        obj = to_int(begin.get("rltools_model_object"), to_int(summary.get("object_bytes")))
        return {
            "model_size_kind": "static_state/model_object",
            "model_state_bytes": state,
            "arena_or_required_memory": state,
            "object_or_control_bytes": obj,
        }
    raise ValueError(f"unknown variant: {variant}")


def build_variant_row(project_root: Path,
                      build_config: str,
                      appli_project_name: str,
                      input_features: int,
                      config: str,
                      variant: str) -> dict[str, object]:
    suffix = config_suffix(input_features, config)
    build_dir = variant_build_dir(project_root, build_config, input_features, config, variant)
    serial_path = build_dir / "serial.log"
    size_path = find_one(build_dir, "*.size.txt")
    elf_path = build_dir / f"{appli_project_name}_{suffix}_{variant}.elf"
    if not elf_path.exists():
        elf_path = find_one(build_dir, "*.elf") or elf_path

    serial = parse_serial_log(serial_path)
    begin = serial["begin"]
    done = serial["done"]
    summaries = serial["summaries"]
    result_seeds = serial["result_seeds"]
    summary = summaries.get(variant, {})
    seed_set = result_seeds.get(variant, set())
    seeds = to_int(done.get("seeds"), to_int(summary.get("seeds")))
    if seeds == 0:
        seeds = len(seed_set)
    size_values = parse_size_file(size_path)

    row: dict[str, object] = {
        "config": config,
        "input_features": to_int(begin.get("input_features"), input_features),
        "variant": variant,
        "seeds": seeds,
        "params": to_int(begin.get("params"), to_int(summary.get("params"))),
        "timing": begin.get("timing", ""),
        "profile_schema": begin.get("profile_schema", ""),
        "optimizer": begin.get("optimizer", ""),
        "batch": to_int(begin.get("batch"), to_int(summary.get("batch"))),
        "rollout": to_int(begin.get("rollout"), to_int(summary.get("rollout"))),
        "epochs": to_int(begin.get("epochs"), to_int(summary.get("epochs"))),
        "optimizer_steps": to_int(
            begin.get("optimizer_steps"), to_int(summary.get("optimizer_steps"))
        ),
        "sample_passes": to_int(
            begin.get("sample_passes"), to_int(summary.get("sample_passes"))
        ),
        "warmups": to_int(begin.get("warmups"), to_int(summary.get("warmups"))),
        "trace_seed": to_int(begin.get("trace_seed")),
        "cycles_avg": to_int(summary.get("cycles_avg")),
        "cycles_min": to_int(summary.get("cycles_min")),
        "cycles_max": to_int(summary.get("cycles_max")),
        "runtime_status": to_int(summary.get("status"), -999),
        "done_status": to_int(done.get("status"), -999),
        "log_exists": 1 if serial["log_exists"] else 0,
        "log_path": portable_project_path(serial_path, project_root),
        "size_path": portable_project_path(size_path, project_root),
        "elf_path": portable_project_path(elf_path, project_root),
        "elf_file_bytes": elf_path.stat().st_size if elf_path.exists() else 0,
    }
    row.update(profile_values(variant, summary))
    row.update(model_size_values(variant, begin, summary))
    row.update(size_values)
    return row


def row_for_variant(rows: dict[str, dict[str, object]], variant: str) -> dict[str, object]:
    return rows.get(variant, {})


def first_value(rows: dict[str, dict[str, object]], key: str, default: object = 0) -> object:
    for variant in VARIANTS:
        value = rows.get(variant, {}).get(key)
        if value not in (None, "", 0):
            return value
    return default


def build_summary_row(config: str,
                      variant_rows: dict[str, dict[str, object]],
                      active_variants: tuple[str, ...]) -> dict[str, object]:
    legacy = row_for_variant(variant_rows, "legacy_c")
    direct = row_for_variant(variant_rows, "cpp_direct_c_backend")
    m55 = row_for_variant(variant_rows, "cpp_m55")
    generic = row_for_variant(variant_rows, "cpp_generic")
    rltools = row_for_variant(variant_rows, "rltools_generic")

    legacy_cycles = to_int(legacy.get("cycles_avg"))
    direct_cycles = to_int(direct.get("cycles_avg"))
    m55_cycles = to_int(m55.get("cycles_avg"))
    generic_cycles = to_int(generic.get("cycles_avg"))
    rltools_cycles = to_int(rltools.get("cycles_avg"))

    row: dict[str, object] = {
        "config": config,
        "input_features": first_value(variant_rows, "input_features"),
        "seeds": first_value(variant_rows, "seeds"),
        "params": first_value(variant_rows, "params"),
        "timing": first_value(variant_rows, "timing", ""),
        "profile_schema": first_value(variant_rows, "profile_schema", ""),
        "optimizer": first_value(variant_rows, "optimizer", ""),
        "batch": first_value(variant_rows, "batch"),
        "rollout": first_value(variant_rows, "rollout"),
        "epochs": first_value(variant_rows, "epochs"),
        "optimizer_steps": first_value(variant_rows, "optimizer_steps"),
        "sample_passes": first_value(variant_rows, "sample_passes"),
        "warmups": first_value(variant_rows, "warmups"),
        "trace_seed": first_value(variant_rows, "trace_seed"),
        "legacy_c_cycles_avg": legacy_cycles,
        "legacy_c_cycles_min": to_int(legacy.get("cycles_min")),
        "legacy_c_cycles_max": to_int(legacy.get("cycles_max")),
        "cpp_direct_c_backend_cycles_avg": direct_cycles,
        "cpp_direct_c_backend_cycles_min": to_int(direct.get("cycles_min")),
        "cpp_direct_c_backend_cycles_max": to_int(direct.get("cycles_max")),
        "direct_over_c": ratio(direct_cycles, legacy_cycles),
        "cpp_m55_cycles_avg": m55_cycles,
        "cpp_m55_cycles_min": to_int(m55.get("cycles_min")),
        "cpp_m55_cycles_max": to_int(m55.get("cycles_max")),
        "m55_over_c": ratio(m55_cycles, legacy_cycles),
        "cpp_generic_cycles_avg": generic_cycles,
        "cpp_generic_cycles_min": to_int(generic.get("cycles_min")),
        "cpp_generic_cycles_max": to_int(generic.get("cycles_max")),
        "generic_over_c": ratio(generic_cycles, legacy_cycles),
        "rltools_generic_cycles_avg": rltools_cycles,
        "rltools_generic_cycles_min": to_int(rltools.get("cycles_min")),
        "rltools_generic_cycles_max": to_int(rltools.get("cycles_max")),
        "rltools_over_c": ratio(rltools_cycles, legacy_cycles),
        "legacy_c_arena_bytes": to_int(legacy.get("arena_or_required_memory")),
        "legacy_c_control_bytes": to_int(legacy.get("object_or_control_bytes")),
        "cpp_direct_required_memory": to_int(direct.get("arena_or_required_memory")),
        "cpp_direct_model_object": to_int(direct.get("object_or_control_bytes")),
        "cpp_m55_required_memory": to_int(m55.get("arena_or_required_memory")),
        "cpp_m55_model_object": to_int(m55.get("object_or_control_bytes")),
        "cpp_generic_required_memory": to_int(generic.get("arena_or_required_memory")),
        "cpp_generic_model_object": to_int(generic.get("object_or_control_bytes")),
        "rltools_static_state": to_int(rltools.get("arena_or_required_memory")),
        "rltools_model_object": to_int(rltools.get("object_or_control_bytes")),
    }

    for variant in VARIANTS:
        source = row_for_variant(variant_rows, variant)
        row[f"{variant}_seeds"] = to_int(source.get("seeds"))
        row[f"{variant}_runtime_status"] = to_int(source.get("runtime_status"), -999)
        row[f"{variant}_done_status"] = to_int(source.get("done_status"), -999)
        row[f"{variant}_model_size_kind"] = source.get("model_size_kind", "")
        row[f"{variant}_model_state_bytes"] = to_int(source.get("model_state_bytes"))
        row[f"{variant}_arena_or_required_memory"] = to_int(source.get("arena_or_required_memory"))
        row[f"{variant}_object_or_control_bytes"] = to_int(source.get("object_or_control_bytes"))
        row[f"{variant}_elf_text"] = to_int(source.get("elf_text"))
        row[f"{variant}_elf_data"] = to_int(source.get("elf_data"))
        row[f"{variant}_elf_bss"] = to_int(source.get("elf_bss"))
        row[f"{variant}_elf_dec"] = to_int(source.get("elf_dec"))
        row[f"{variant}_elf_file_bytes"] = to_int(source.get("elf_file_bytes"))
        row[f"{variant}_elf_dec_over_c"] = ratio(
            to_int(source.get("elf_dec")), to_int(legacy.get("elf_dec"))
        )
        row[f"{variant}_log_path"] = source.get("log_path", "")
        row[f"{variant}_size_path"] = source.get("size_path", "")
        row[f"{variant}_elf_path"] = source.get("elf_path", "")

    for variant in VARIANTS:
        source = row_for_variant(variant_rows, variant)
        row.update(profile_values(variant, {}))
        for key, value in source.items():
            if str(key).startswith(f"{variant}_profile_"):
                row[key] = value

    row["status"] = 0 if all(
        to_int(row_for_variant(variant_rows, variant).get("runtime_status"), -999) == 0
        and to_int(row_for_variant(variant_rows, variant).get("done_status"), -999) == 0
        and to_int(row_for_variant(variant_rows, variant).get("cycles_avg")) > 0
        for variant in active_variants
    ) else -1
    return row


def fmt_int(value: object) -> str:
    return f"{int(value):,}"


def fmt_ratio(value: object) -> str:
    return f"{float(value):.3f}" if float(value) else "0.000"


def model_cell(row: dict[str, object], variant: str) -> str:
    if variant == "legacy_c":
        return fmt_int(
            to_int(row.get("legacy_c_arena_bytes")) + to_int(row.get("legacy_c_control_bytes"))
        )
    return f"{fmt_int(row[f'{variant}_arena_or_required_memory'])}/{fmt_int(row[f'{variant}_object_or_control_bytes'])}"


def elf_cell(row: dict[str, object], variant: str) -> str:
    return (
        f"{fmt_int(row[f'{variant}_elf_dec'])}/"
        f"{fmt_int(row[f'{variant}_elf_file_bytes'])} "
        f"({fmt_ratio(row[f'{variant}_elf_dec_over_c'])}x)"
    )


def elf_cell_plain(row: dict[str, object], variant: str) -> str:
    return f"{fmt_int(row[f'{variant}_elf_dec'])}/{fmt_int(row[f'{variant}_elf_file_bytes'])}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
        "input_features",
        "seeds",
        "params",
        "timing",
        "profile_schema",
        "optimizer",
        "batch",
        "rollout",
        "epochs",
        "optimizer_steps",
        "sample_passes",
        "warmups",
        "trace_seed",
        "legacy_c_cycles_avg",
        "legacy_c_cycles_min",
        "legacy_c_cycles_max",
        "cpp_direct_c_backend_cycles_avg",
        "cpp_direct_c_backend_cycles_min",
        "cpp_direct_c_backend_cycles_max",
        "direct_over_c",
        "cpp_m55_cycles_avg",
        "cpp_m55_cycles_min",
        "cpp_m55_cycles_max",
        "m55_over_c",
        "cpp_generic_cycles_avg",
        "cpp_generic_cycles_min",
        "cpp_generic_cycles_max",
        "generic_over_c",
        "rltools_generic_cycles_avg",
        "rltools_generic_cycles_min",
        "rltools_generic_cycles_max",
        "rltools_over_c",
        *[f"{variant}_profile_cycles_avg" for variant in VARIANTS],
        *[
            f"{variant}_profile_{component}_avg"
            for variant in VARIANTS
            for component in PROFILE_COMPONENTS
        ],
        "legacy_c_arena_bytes",
        "legacy_c_control_bytes",
        "cpp_direct_required_memory",
        "cpp_direct_model_object",
        "cpp_m55_required_memory",
        "cpp_m55_model_object",
        "cpp_generic_required_memory",
        "cpp_generic_model_object",
        "rltools_static_state",
        "rltools_model_object",
        *[
            field
            for variant in VARIANTS
            for field in (
                f"{variant}_seeds",
                f"{variant}_runtime_status",
                f"{variant}_done_status",
                f"{variant}_model_size_kind",
                f"{variant}_model_state_bytes",
                f"{variant}_arena_or_required_memory",
                f"{variant}_object_or_control_bytes",
                f"{variant}_elf_text",
                f"{variant}_elf_data",
                f"{variant}_elf_bss",
                f"{variant}_elf_dec",
                f"{variant}_elf_file_bytes",
                f"{variant}_elf_dec_over_c",
                f"{variant}_log_path",
                f"{variant}_size_path",
                f"{variant}_elf_path",
            )
        ],
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path,
                   rows: list[dict[str, object]],
                   input_features: int,
                   active_variants: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_done = all(int(row["status"]) == 0 for row in rows)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# STM32N6 EL_C_vsCpp Per-Variant Sweep - {dt.date.today().isoformat()} - 10 seeds\n\n")
        f.write("Board target: STM32N6 Cortex-M55 with MVE.\n")
        f.write(f"Task: deterministic linear regression, input {input_features}, output 1, batch 256.\n")
        f.write("Build/run unit: one firmware ELF per variant and per network size.\n")
        f.write("Protocol: Adam, rollout 1024, 2 epochs, 8 optimizer steps, 2048 sample-passes per measured run.\n")
        f.write("Batch semantics: EdgeLearning++ accumulates gradients over 256 samples before one Adam update; RLTools uses a static tensor with shape `[256, input_features]` and one forward/loss/backward/update per minibatch.\n")
        f.write("Warm-up: 2 full training runs per seed, with model and optimizer reset before the measured run.\n")
        f.write("Timing: pre-generated rollout hot path only; setup, import/export, reset, sample generation, warm-up, traces, and serial I/O are outside DWT.\n")
        f.write("Profiling: training-loop component counters are collected in a separate equivalent pass with the same initial parameters and dataset, then averaged over seeds.\n")
        if "legacy_c" in active_variants:
            f.write("Legacy C exposes `sample_train_step` as one combined forward/loss/backward component because those operations are encapsulated by the C API.\n")
        f.write("Convergence trace: seed 0, minibatch MSE after each Adam update, emitted by an untimed diagnostic pass.\n")
        if "legacy_c" in active_variants:
            f.write("Build: static C arena and static C++ model, all firmware objects compiled with `-Ofast`.\n")
        else:
            f.write("Build: static C++/RLTools model storage, all firmware objects compiled with `-Ofast`.\n")
        f.write("ELF size columns are from the same per-variant image used for the runtime row.\n\n")
        if "legacy_c" in active_variants:
            f.write("This report includes private legacy-C ablation rows from an external checkout.\n\n")
        else:
            f.write("This report uses the public C++/RLTools variant set and does not require the legacy C checkout.\n\n")
        if all_done:
            f.write("All per-variant runs completed with `DONE status=0`.\n\n")
        else:
            f.write("At least one per-variant run did not complete with `DONE status=0`.\n\n")
        if "legacy_c" in active_variants:
            f.write(
                "| Config | Input | Seeds | Warm-ups | Params | C M55 avg | Direct C-backend avg | Direct/C | "
                "C++ M55 avg | M55/C | C++ Generic avg | Generic/C | RLTools Batch avg | RLTools/C |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in rows:
                f.write(
                    f"| {row['config']} | {row['input_features']} | {row['seeds']} | {row['warmups']} | {row['params']} | "
                    f"{row['legacy_c_cycles_avg']} | {row['cpp_direct_c_backend_cycles_avg']} | "
                    f"{fmt_ratio(row['direct_over_c'])} | {row['cpp_m55_cycles_avg']} | "
                    f"{fmt_ratio(row['m55_over_c'])} | {row['cpp_generic_cycles_avg']} | "
                    f"{fmt_ratio(row['generic_over_c'])} | {row['rltools_generic_cycles_avg']} | "
                    f"{fmt_ratio(row['rltools_over_c'])} |\n"
                )
            f.write("\n")
            f.write(
                "| Config | C model | Direct req/obj | M55 req/obj | Generic req/obj | "
                "RLTools state/obj | C ELF dec/file | Direct ELF dec/file | M55 ELF dec/file | "
                "Generic ELF dec/file | RLTools ELF dec/file |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in rows:
                f.write(
                    f"| {row['config']} | {model_cell(row, 'legacy_c')} | "
                    f"{model_cell(row, 'cpp_direct_c_backend')} | {model_cell(row, 'cpp_m55')} | "
                    f"{model_cell(row, 'cpp_generic')} | {model_cell(row, 'rltools_generic')} | "
                    f"{elf_cell(row, 'legacy_c')} | {elf_cell(row, 'cpp_direct_c_backend')} | "
                    f"{elf_cell(row, 'cpp_m55')} | {elf_cell(row, 'cpp_generic')} | "
                    f"{elf_cell(row, 'rltools_generic')} |\n"
                )
        else:
            f.write(
                "| Config | Input | Seeds | Warm-ups | Params | C++ M55 avg | C++ Generic avg | "
                "Generic/M55 | RLTools Batch avg | RLTools/M55 | RLTools/M55 runtime ratio |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in rows:
                m55 = to_int(row["cpp_m55_cycles_avg"])
                generic = to_int(row["cpp_generic_cycles_avg"])
                rltools = to_int(row["rltools_generic_cycles_avg"])
                f.write(
                    f"| {row['config']} | {row['input_features']} | {row['seeds']} | {row['warmups']} | {row['params']} | "
                    f"{m55} | {generic} | {fmt_ratio(ratio(generic, m55))} | "
                    f"{rltools} | {fmt_ratio(ratio(rltools, m55))} | "
                    f"{fmt_ratio(ratio(rltools, m55))}x |\n"
                )
            f.write("\n")
            f.write(
                "| Config | M55 req/obj | Generic req/obj | RLTools state/obj | "
                "M55 ELF dec/file | Generic ELF dec/file | RLTools ELF dec/file |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for row in rows:
                f.write(
                    f"| {row['config']} | {model_cell(row, 'cpp_m55')} | "
                    f"{model_cell(row, 'cpp_generic')} | {model_cell(row, 'rltools_generic')} | "
                    f"{elf_cell_plain(row, 'cpp_m55')} | {elf_cell_plain(row, 'cpp_generic')} | "
                    f"{elf_cell_plain(row, 'rltools_generic')} |\n"
                )
        f.write("\n")
        f.write("Raw UART logs, `.size.txt` files, and ELF paths are referenced in the CSV.\n")


def main() -> int:
    args = parse_args()
    env = load_env(args.env_file)
    active_variants = resolve_variants(args, env)
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

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]
    output_stem = args.output_stem or (
        repo_root
        / "benchmarks"
        / "firmware"
        / "stm32n6"
        / "el_cvscpp_ablation"
        / "results"
        / f"stm32n6_sweep_{dt.date.today().isoformat()}_input{input_features}_10seed"
    )

    rows: list[dict[str, object]] = []
    for config in args.configs:
        variant_rows = {
            variant: build_variant_row(
                project_root,
                build_config,
                appli_project_name,
                input_features,
                config,
                variant,
            )
            for variant in active_variants
        }
        rows.append(build_summary_row(config, variant_rows, active_variants))

    write_csv(output_stem.with_suffix(".csv"), rows)
    write_markdown(output_stem.with_suffix(".md"), rows, input_features, active_variants)
    print(output_stem.with_suffix(".csv"))
    print(output_stem.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
