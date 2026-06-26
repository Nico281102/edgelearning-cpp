#!/usr/bin/env python3
"""Build and report host code size for EdgeLearning C++ and optional old C baseline.

The old C source is never copied into edgelearning-cpp. If a baseline checkout is
provided, this script writes a temporary harness outside the repository, compiles
against that checkout, records section sizes, and deletes the temporary build
directory when the process exits.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


BASELINE_COMMIT = "0085814908ca1b57ece4fe367361d084fd74aa3e"


C_BASELINE_HARNESS = r'''
#include "edgelearning.h"

#define ARENA_SIZE (1024 * 10)

static volatile el_float_t sink = 0.0f;

int main(void) {
    EL_ALIGNED(EL_ALIGNMENT_BYTES) uint8_t memory_arena[ARENA_SIZE];
    EL_ALIGNED(EL_ALIGNMENT_BYTES) el_float_t reg_features_data[] = { 0, 1, 2, 3 };
    EL_ALIGNED(EL_ALIGNMENT_BYTES) el_float_t reg_targets_data[]  = { 1, 3, 5, 7 };

    el_context_t ctx;
    if (el_ctx_init(&ctx, memory_arena, ARENA_SIZE) != EL_OK) return 1;

    el_network_t net;
    if (el_network_init(&ctx, &net, 2) != EL_OK) return 2;
    if (el_network_add_dense(&net, 1, 4, EL_ACT_TANH, true, false) == 0) return 3;
    if (el_network_add_dense(&net, 4, 1, EL_ACT_NONE, true, false) == 0) return 4;

    el_tensor_t features = { .data = reg_features_data, .rows = 4, .cols = 1 };
    el_tensor_t targets  = { .data = reg_targets_data,  .rows = 4, .cols = 1 };
    el_float_t loss = el_supervised_train(&net, &features, &targets, 1, 0.05f, 4,
                                          EL_LOSS_MSE, 0);

    el_float_t out_buf[1];
    el_tensor_t out = { .data = out_buf, .rows = 1, .cols = 1 };
    el_tensor_t sample = { .data = &reg_features_data[1], .rows = 1, .cols = 1 };
    el_supervised_predict(&net, &sample, &out);
    sink = loss + out_buf[0];
    return 0;
}
'''


@dataclass
class SizeResult:
    implementation: str
    binary: Path
    compiler: str
    flags: str
    commit: str
    text: int
    rodata: int
    data: int
    bss: int
    other: int
    total: int
    file_bytes: int
    sections: dict[str, int]


def run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        joined = " ".join(cmd)
        raise RuntimeError(f"command failed ({joined})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result.stdout.strip()


def find_llvm_size() -> str:
    for candidate in ("llvm-size",):
        path = shutil.which(candidate)
        if path:
            return path
    xcrun = shutil.which("xcrun")
    if xcrun:
        try:
            path = run([xcrun, "--find", "llvm-size"])
            if path:
                return path
        except RuntimeError:
            pass
    path = shutil.which("size")
    if path:
        return path
    raise RuntimeError("could not find llvm-size or size")


def parse_sysv_size(size_tool: str, binary: Path) -> dict[str, int]:
    output = run([size_tool, "--format=sysv", str(binary)])
    sections: dict[str, int] = {}
    in_table = False
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "section" and parts[1] == "size":
            in_table = True
            continue
        if not in_table:
            continue
        if len(parts) >= 2 and parts[0] == "Total":
            sections["Total"] = int(parts[1])
            continue
        if len(parts) >= 3:
            try:
                sections[parts[0]] = int(parts[1])
            except ValueError:
                pass
    if "Total" not in sections:
        raise RuntimeError(f"could not parse total size from {binary}")
    return sections


def categorize(sections: dict[str, int]) -> tuple[int, int, int, int, int, int]:
    text = rodata = data = bss = 0
    for name, size in sections.items():
        lowered = name.lower()
        if name == "Total":
            continue
        if "text" in lowered or "stubs" in lowered or lowered in {".init", ".fini"}:
            text += size
        elif "rodata" in lowered or "const" in lowered or "cstring" in lowered:
            rodata += size
        elif "bss" in lowered or "common" in lowered or "zerofill" in lowered:
            bss += size
        elif "data" in lowered or "got" in lowered or "la_symbol_ptr" in lowered:
            data += size
    total = sections["Total"]
    other = total - text - rodata - data - bss
    return text, rodata, data, bss, other, total


def compiler_version(compiler: str) -> str:
    try:
        first_line = run([compiler, "--version"]).splitlines()[0]
        return first_line
    except Exception:
        return compiler


def cpp_commit(repo: Path) -> str:
    try:
        commit = run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"])
        dirty = run([
            "git",
            "-C",
            str(repo),
            "status",
            "--short",
            "--",
            ".",
            ":!benchmarks/results",
        ])
        return f"{commit} (dirty)" if dirty else commit
    except Exception:
        return "unknown"


def baseline_commit(path: Path) -> str:
    try:
        return run(["git", "-C", str(path), "rev-parse", "HEAD"])
    except Exception:
        return "unknown"


def baseline_sources(path: Path) -> list[Path]:
    roots = [path / "0_Common", path / "1_Engine", path / "2_Network", path / "3_Paradigms"]
    sources: list[Path] = []
    excluded = {
        "el_backend_m55.c",
        "el_backend_CMSIS_DSP.c",
        "el_backend_m55_rsqrt_test.c",
        "el_backend_m55_rsqrt_fast.c",
        "el_backend_m55_rsqrt_test_2.c",
        "el_backend_m55_acc2_test.c",
        "el_backend_m55_acc8_test.c",
        "el_backend_general_cpu_zero_fast.c",
        "el_platform_stm32n657xx.c",
    }
    for root in roots:
        for source in sorted(root.glob("*.c")):
            if source.name not in excluded:
                sources.append(source)
    return sources


def link_options() -> list[str]:
    if sys.platform == "darwin":
        return ["-Wl,-dead_strip"]
    return ["-Wl,--gc-sections"]


def measure_binary(
    implementation: str,
    binary: Path,
    compiler: str,
    flags: str,
    commit: str,
    size_tool: str,
) -> SizeResult:
    sections = parse_sysv_size(size_tool, binary)
    text, rodata, data, bss, other, total = categorize(sections)
    return SizeResult(
        implementation=implementation,
        binary=binary,
        compiler=compiler,
        flags=flags,
        commit=commit,
        text=text,
        rodata=rodata,
        data=data,
        bss=bss,
        other=other,
        total=total,
        file_bytes=binary.stat().st_size,
        sections=sections,
    )


def build_c_baseline(path: Path, cc: str, cflags: list[str], size_tool: str) -> SizeResult:
    commit = baseline_commit(path)
    if commit != "unknown" and commit != BASELINE_COMMIT:
        print(
            f"warning: C baseline checkout is {commit}, expected {BASELINE_COMMIT}",
            file=sys.stderr,
        )

    with tempfile.TemporaryDirectory(prefix="edgelearning_c_code_size_") as temp:
        temp_dir = Path(temp)
        harness = temp_dir / "c_code_size_regression.c"
        binary = temp_dir / "c_code_size_regression"
        harness.write_text(C_BASELINE_HARNESS)

        flags = [
            "-Os",
            "-ffunction-sections",
            "-fdata-sections",
            "-I",
            str(path),
            "-I",
            str(path / "Inc"),
            *cflags,
        ]
        cmd = [
            cc,
            *flags,
            str(harness),
            *[str(source) for source in baseline_sources(path)],
            "-o",
            str(binary),
            *link_options(),
            "-lm",
        ]
        run(cmd)
        return measure_binary(
            "old_c_baseline",
            binary,
            compiler_version(cc),
            " ".join(flags + link_options() + ["-lm"]),
            commit,
            size_tool,
        )


def write_reports(results: list[SizeResult], result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "code_size_report.csv"
    md_path = result_dir / "code_size_report.md"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "implementation",
            "commit",
            "text_bytes",
            "rodata_bytes",
            "data_bytes",
            "bss_bytes",
            "other_section_bytes",
            "total_section_bytes",
            "file_bytes",
            "compiler",
            "flags",
            "binary",
        ])
        for r in results:
            writer.writerow([
                r.implementation,
                r.commit,
                r.text,
                r.rodata,
                r.data,
                r.bss,
                r.other,
                r.total,
                r.file_bytes,
                r.compiler,
                r.flags,
                r.binary,
            ])

    with md_path.open("w") as f:
        f.write("# Code Size Report\n\n")
        f.write("Baseline C commit: ")
        f.write(f"`{BASELINE_COMMIT}`.\n\n")
        f.write("The old C source is not vendored, copied, or published in this repository. ")
        f.write("When `EDGE_C_BASELINE_DIR` or `--c-baseline-dir` is provided, the script ")
        f.write("compiles a temporary harness outside this repository and records section sizes only.\n\n")
        f.write("## Methodology\n\n")
        f.write("- Topology: `1 -> Dense<4, Tanh> -> Dense<1, Linear>`\n")
        f.write("- Training path: one supervised sample step plus flush/update\n")
        f.write("- Optimization goal: code size, using `-Os`, function/data sections, and dead-code stripping\n")
        f.write("- Reported `total_section_bytes` is the sum of object sections from `llvm-size --format=sysv`; ")
        f.write("it is not the page-aligned executable file size.\n\n")
        f.write("## Results\n\n")
        f.write("| Implementation | Commit | Text | RoData | Data | BSS | Other | Total sections | File bytes |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| {r.implementation} | `{r.commit}` | {r.text} | {r.rodata} | "
                f"{r.data} | {r.bss} | {r.other} | {r.total} | {r.file_bytes} |\n"
            )
        f.write("\n## Raw Sections\n\n")
        for r in results:
            f.write(f"### {r.implementation}\n\n")
            f.write(f"- Compiler: `{r.compiler}`\n")
            f.write(f"- Flags: `{r.flags}`\n")
            f.write(f"- Binary measured: `{r.binary}`\n\n")
            f.write("```text\n")
            for name, size in r.sections.items():
                f.write(f"{name:24s} {size}\n")
            f.write("```\n\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-bin", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--c-baseline-dir", type=Path, default=None)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--cflag", action="append", default=[])
    args = parser.parse_args()

    size_tool = find_llvm_size()
    results = [
        measure_binary(
            "edgelearning_cpp",
            args.cpp_bin,
            compiler_version(os.environ.get("CXX", "c++")),
            "-Os -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti "
            + " ".join(link_options()),
            cpp_commit(args.repo),
            size_tool,
        )
    ]

    baseline_dir = args.c_baseline_dir or (
        Path(os.environ["EDGE_C_BASELINE_DIR"]) if os.environ.get("EDGE_C_BASELINE_DIR") else None
    )
    if baseline_dir:
        results.append(build_c_baseline(baseline_dir, args.cc, args.cflag, size_tool))
    else:
        print("No C baseline path provided; set EDGE_C_BASELINE_DIR or pass --c-baseline-dir.")

    write_reports(results, args.result_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
