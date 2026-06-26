#!/usr/bin/env python3
"""Build CSV and Markdown reports from EL_C_vsCpp STM32N6 sweep artifacts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path


DEFAULT_CONFIGS = ("8x8", "16x8", "16x16", "32x16", "32x32", "64x32")
VARIANTS = ("legacy_c", "cpp_direct_c_backend", "cpp_m55", "cpp_generic", "rltools_generic")
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
COMPARES = {
    "legacy_c_vs_cpp_direct_c_backend": "compare_direct_ok",
    "legacy_c_vs_cpp_m55": "compare_m55_ok",
    "legacy_c_vs_cpp_generic": "compare_generic_ok",
    "cpp_generic_vs_rltools_generic": "compare_rltools_ok",
}


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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def parse_kv_line(line: str) -> tuple[str, dict[str, str]]:
    parts = line.strip().split()
    if not parts:
        return "", {}
    kind = parts[0]
    values: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return kind, values


def to_int(value: str | None, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def to_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def parse_serial_log(path: Path) -> dict[str, object]:
    begin: dict[str, str] = {}
    done: dict[str, str] = {}
    summaries: dict[str, dict[str, str]] = {}
    compare_ok: dict[str, list[int]] = {name: [] for name in COMPARES}
    result_seeds: dict[str, set[int]] = {variant: set() for variant in VARIANTS}

    if not path.exists():
        return {
            "begin": begin,
            "done": done,
            "summaries": summaries,
            "compare_ok": compare_ok,
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
        elif kind == "COMPARE":
            name = values.get("name")
            if name in compare_ok:
                compare_ok[name].append(to_int(values.get("ok")))
        elif kind == "RESULT":
            variant = values.get("variant")
            seed = values.get("seed")
            if variant in result_seeds and seed is not None:
                result_seeds[variant].add(int(seed))

    return {
        "begin": begin,
        "done": done,
        "summaries": summaries,
        "compare_ok": compare_ok,
        "result_seeds": result_seeds,
        "log_exists": True,
    }


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


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def profile_values(prefix: str, summary: dict[str, str]) -> dict[str, int]:
    values = {
        f"{prefix}_profile_{component}_avg": to_int(summary.get(f"prof_{component}_avg"))
        for component in PROFILE_COMPONENTS
    }
    values[f"{prefix}_profile_cycles_avg"] = to_int(summary.get("prof_cycles_avg"))
    return values


def all_ok(values: list[int]) -> int:
    return 1 if values and all(value == 1 for value in values) else 0


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


def build_row(project_root: Path,
              build_config: str,
              appli_project_name: str,
              input_features: int,
              config: str) -> dict[str, object]:
    h1, h2 = config.split("x", 1)
    config_suffix = f"{input_features}_{h1}x{h2}_1"
    build_dir = project_root / "STM32CubeIDE" / "Appli" / build_config / config_suffix
    serial_path = build_dir / "serial.log"
    size_path = find_one(build_dir, "*.size.txt")
    elf_path = build_dir / f"{appli_project_name}_{config_suffix}.elf"
    if not elf_path.exists():
        elf_path = find_one(build_dir, "*.elf") or elf_path

    serial = parse_serial_log(serial_path)
    begin = serial["begin"]
    done = serial["done"]
    summaries = serial["summaries"]
    compare_ok = serial["compare_ok"]
    result_seeds = serial["result_seeds"]
    size_values = parse_size_file(size_path)

    legacy = summaries.get("legacy_c", {})
    direct = summaries.get("cpp_direct_c_backend", {})
    m55 = summaries.get("cpp_m55", {})
    generic = summaries.get("cpp_generic", {})
    rltools = summaries.get("rltools_generic", {})

    legacy_cycles = to_int(legacy.get("cycles_avg"))
    direct_cycles = to_int(direct.get("cycles_avg"))
    m55_cycles = to_int(m55.get("cycles_avg"))
    generic_cycles = to_int(generic.get("cycles_avg"))
    rltools_cycles = to_int(rltools.get("cycles_avg"))

    seeds = to_int(done.get("seeds"), to_int(begin.get("seeds")))
    if seeds == 0:
        seeds = max((len(seeds_set) for seeds_set in result_seeds.values()), default=0)

    row: dict[str, object] = {
        "config": config,
        "input_features": to_int(begin.get("input_features"), input_features),
        "seeds": seeds,
        "params": to_int(begin.get("params"), to_int(legacy.get("params"))),
        "timing": begin.get("timing", ""),
        "profile_schema": begin.get("profile_schema", ""),
        "optimizer": begin.get("optimizer", ""),
        "batch": to_int(begin.get("batch"), to_int(legacy.get("batch"))),
        "rollout": to_int(begin.get("rollout"), to_int(legacy.get("rollout"))),
        "epochs": to_int(begin.get("epochs"), to_int(legacy.get("epochs"))),
        "optimizer_steps": to_int(
            begin.get("optimizer_steps"), to_int(legacy.get("optimizer_steps"))
        ),
        "sample_passes": to_int(
            begin.get("sample_passes"), to_int(legacy.get("sample_passes"))
        ),
        "warmups": to_int(begin.get("warmups"), to_int(legacy.get("warmups"))),
        "trace_seed": to_int(begin.get("trace_seed")),
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
        "legacy_c_arena_bytes": to_int(begin.get("legacy_c_arena"), to_int(legacy.get("arena_bytes"))),
        "legacy_c_control_bytes": to_int(begin.get("legacy_c_control"), to_int(legacy.get("object_bytes"))),
        "cpp_direct_required_memory": to_int(
            begin.get("cpp_direct_required_memory"), to_int(direct.get("arena_bytes"))
        ),
        "cpp_direct_model_object": to_int(
            begin.get("cpp_direct_model_object"), to_int(direct.get("object_bytes"))
        ),
        "cpp_m55_required_memory": to_int(
            begin.get("cpp_m55_required_memory"), to_int(m55.get("arena_bytes"))
        ),
        "cpp_m55_model_object": to_int(
            begin.get("cpp_m55_model_object"), to_int(m55.get("object_bytes"))
        ),
        "cpp_generic_required_memory": to_int(
            begin.get("cpp_generic_required_memory"), to_int(generic.get("arena_bytes"))
        ),
        "cpp_generic_model_object": to_int(
            begin.get("cpp_generic_model_object"), to_int(generic.get("object_bytes"))
        ),
        "rltools_static_state": to_int(
            begin.get("rltools_static_state"), to_int(rltools.get("arena_bytes"))
        ),
        "rltools_model_object": to_int(
            begin.get("rltools_model_object"), to_int(rltools.get("object_bytes"))
        ),
        "status": to_int(done.get("status"), -999),
        "compare_direct_ok": all_ok(compare_ok["legacy_c_vs_cpp_direct_c_backend"]),
        "compare_m55_ok": all_ok(compare_ok["legacy_c_vs_cpp_m55"]),
        "compare_generic_ok": all_ok(compare_ok["legacy_c_vs_cpp_generic"]),
        "compare_rltools_ok": all_ok(compare_ok["cpp_generic_vs_rltools_generic"]),
        "log_path": portable_project_path(serial_path, project_root),
        "size_path": portable_project_path(size_path, project_root),
        "elf_path": portable_project_path(elf_path, project_root),
        "elf_file_bytes": elf_path.stat().st_size if elf_path.exists() else 0,
    }
    row.update(profile_values("legacy_c", legacy))
    row.update(profile_values("cpp_direct_c_backend", direct))
    row.update(profile_values("cpp_m55", m55))
    row.update(profile_values("cpp_generic", generic))
    row.update(profile_values("rltools_generic", rltools))
    row.update(size_values)
    return row


def fmt_ratio(value: object) -> str:
    return f"{float(value):.3f}" if float(value) else "0.000"


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
        "elf_text",
        "elf_data",
        "elf_bss",
        "elf_dec",
        "elf_file_bytes",
        "status",
        "compare_direct_ok",
        "compare_m55_ok",
        "compare_generic_ok",
        "compare_rltools_ok",
        "log_path",
        "size_path",
        "elf_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], input_features: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_done = all(int(row["status"]) == 0 for row in rows)
    all_compares = all(
        int(row["compare_direct_ok"]) == 1
        and int(row["compare_m55_ok"]) == 1
        and int(row["compare_generic_ok"]) == 1
        and int(row["compare_rltools_ok"]) == 1
        for row in rows
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# STM32N6 EL_C_vsCpp Sweep - {dt.date.today().isoformat()} - 10 seeds\n\n")
        f.write("Board target: STM32N6 Cortex-M55 with MVE.\n")
        f.write(f"Task: deterministic linear regression, input {input_features}, output 1, batch 256.\n")
        f.write("Protocol: Adam, rollout 1024, 2 epochs, 8 optimizer steps, 2048 sample-passes per measured run.\n")
        f.write("Warm-up: 2 full training runs per variant/seed, with model and optimizer reset before the measured run.\n")
        f.write("Timing: pre-generated rollout hot path only; setup, import/export, reset, sample generation, warm-up, traces, and comparisons are outside DWT.\n")
        f.write("Profiling: training-loop component counters are collected in a separate equivalent pass with the same initial parameters and dataset, then averaged over seeds.\n")
        f.write("Legacy C exposes `sample_train_step` as one combined forward/loss/backward component because those operations are encapsulated by the C API.\n")
        f.write("Convergence trace: seed 0, minibatch MSE after each Adam update, emitted by an untimed diagnostic pass.\n")
        f.write("Build: static C arena and static C++ model, all firmware objects compiled with `-Ofast`.\n\n")
        f.write(f"All runs completed with `DONE status=0`: `{int(all_done)}`.\n")
        f.write(f"All numerical comparisons passed for every seed: `{int(all_compares)}`.\n\n")
        f.write(
            "| Config | Input | Seeds | Warm-ups | Params | C M55 avg | Direct C-backend avg | Direct/C | "
            "C++ M55 avg | M55/C | C++ Generic avg | Generic/C | "
            "RLTools Generic avg | RLTools/C | C arena+ctrl | Direct req/obj | "
            "M55 req/obj | Generic req/obj | RLTools state/obj | "
            "ELF text | ELF data | ELF bss | ELF dec | ELF file |\n"
        )
        f.write(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for row in rows:
            c_total = int(row["legacy_c_arena_bytes"]) + int(row["legacy_c_control_bytes"])
            direct_size = f"{row['cpp_direct_required_memory']}/{row['cpp_direct_model_object']}"
            m55_size = f"{row['cpp_m55_required_memory']}/{row['cpp_m55_model_object']}"
            generic_size = f"{row['cpp_generic_required_memory']}/{row['cpp_generic_model_object']}"
            rltools_size = f"{row['rltools_static_state']}/{row['rltools_model_object']}"
            f.write(
                f"| {row['config']} | {row['input_features']} | {row['seeds']} | {row['warmups']} | {row['params']} | "
                f"{row['legacy_c_cycles_avg']} | {row['cpp_direct_c_backend_cycles_avg']} | "
                f"{fmt_ratio(row['direct_over_c'])} | {row['cpp_m55_cycles_avg']} | "
                f"{fmt_ratio(row['m55_over_c'])} | {row['cpp_generic_cycles_avg']} | "
                f"{fmt_ratio(row['generic_over_c'])} | {row['rltools_generic_cycles_avg']} | "
                f"{fmt_ratio(row['rltools_over_c'])} | {c_total} | {direct_size} | "
                f"{m55_size} | {generic_size} | {rltools_size} | "
                f"{row['elf_text']} | {row['elf_data']} | "
                f"{row['elf_bss']} | {row['elf_dec']} | {row['elf_file_bytes']} |\n"
            )
        f.write("\nRaw UART logs and `.size.txt` files are referenced in the CSV.\n")


def main() -> int:
    args = parse_args()
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
        Path(__file__).resolve().parent
        / "results"
        / f"stm32n6_sweep_{dt.date.today().isoformat()}_input{input_features}_10seed"
    )

    rows = [
        build_row(project_root, build_config, appli_project_name, input_features, config)
        for config in args.configs
    ]
    write_csv(output_stem.with_suffix(".csv"), rows)
    write_markdown(output_stem.with_suffix(".md"), rows, input_features)
    print(output_stem.with_suffix(".csv"))
    print(output_stem.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
