#!/usr/bin/env python3
"""Measure ELF section sizes for the static M55 regression sweep binaries."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SizeResult:
    target: str
    case: str
    binary: Path
    file_bytes: int
    text: int
    rodata: int
    data: int
    bss: int
    other: int
    total: int


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def find_size_tool() -> str:
    if os.environ.get("EDGE_SIZE_TOOL"):
        return os.environ["EDGE_SIZE_TOOL"]
    for candidate in ("arm-none-eabi-size", "llvm-size"):
        path = shutil.which(candidate)
        if path:
            return path
    xcrun = shutil.which("xcrun")
    if xcrun:
        try:
            path = run([xcrun, "--find", "llvm-size"]).strip()
            if path:
                return path
        except RuntimeError:
            pass
    path = shutil.which("size")
    if path:
        return path
    raise RuntimeError("could not find a size tool; set EDGE_SIZE_TOOL")


def parse_size_output(output: str) -> dict[str, int]:
    sections: dict[str, int] = {}
    in_table = False
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "section" and parts[1] == "size":
            in_table = True
            continue
        if not in_table and len(parts) >= 2 and parts[0] == "Total":
            sections["Total"] = int(parts[1])
            continue
        if len(parts) >= 2 and parts[0] == "Total":
            sections["Total"] = int(parts[1])
            continue
        if in_table and len(parts) >= 2:
            try:
                sections[parts[0]] = int(parts[1])
            except ValueError:
                pass
    if "Total" not in sections:
        raise RuntimeError("could not parse total section size from size output")
    return sections


def read_sections(size_tool: str, binary: Path) -> dict[str, int]:
    try:
        return parse_size_output(run([size_tool, "--format=sysv", str(binary)]))
    except RuntimeError:
        return parse_size_output(run([size_tool, "-A", str(binary)]))


def categorize(sections: dict[str, int]) -> tuple[int, int, int, int, int, int]:
    text = rodata = data = bss = 0
    for name, size in sections.items():
        lowered = name.lower()
        if name == "Total":
            continue
        if "text" in lowered or lowered in {".init", ".fini"}:
            text += size
        elif "rodata" in lowered or "const" in lowered or "cstring" in lowered:
            rodata += size
        elif "bss" in lowered or "common" in lowered or "zerofill" in lowered:
            bss += size
        elif "data" in lowered or "got" in lowered:
            data += size
    total = sections["Total"]
    other = total - text - rodata - data - bss
    return text, rodata, data, bss, other, total


def case_from_target(target: str) -> str:
    if target.startswith("test_m55_regression_"):
        return target.removeprefix("test_m55_regression_")
    if "_regression_" in target:
        return target.rsplit("_regression_", maxsplit=1)[-1]
    return target


def measure(target: str, binary: Path, size_tool: str) -> SizeResult:
    sections = read_sections(size_tool, binary)
    text, rodata, data, bss, other, total = categorize(sections)
    return SizeResult(
        target=target,
        case=case_from_target(target),
        binary=binary,
        file_bytes=binary.stat().st_size,
        text=text,
        rodata=rodata,
        data=data,
        bss=bss,
        other=other,
        total=total,
    )


def write_reports(results: list[SizeResult], result_dir: Path, size_tool: str) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "m55_regression_elf_size.csv"
    md_path = result_dir / "m55_regression_elf_size.md"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "target",
                "case",
                "text_bytes",
                "rodata_bytes",
                "data_bytes",
                "bss_bytes",
                "other_section_bytes",
                "total_section_bytes",
                "file_bytes",
                "binary",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.target,
                    r.case,
                    r.text,
                    r.rodata,
                    r.data,
                    r.bss,
                    r.other,
                    r.total,
                    r.file_bytes,
                    r.binary,
                ]
            )

    with md_path.open("w") as f:
        f.write("# M55 Regression ELF Size\n\n")
        f.write(f"Size tool: `{size_tool}`.\n\n")
        f.write("Each binary contains one static regression-sweep topology: input 32, two hidden ")
        f.write("Dense/ReLU layers, one linear output neuron, and batch size 256. Static model ")
        f.write("or arena storage is intentionally included so `.bss` reflects the deployment shape.\n\n")
        f.write("Legacy-C targets are split into `direct_backend` (C++ model calling the legacy ")
        f.write("C backend kernels directly) and `native_m55` (C++ model using `Backend::M55`).\n\n")
        f.write("| Target | Case | Text | RoData | Data | BSS | Other | Total sections | File bytes |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| {r.target} | {r.case} | {r.text} | {r.rodata} | {r.data} | "
                f"{r.bss} | {r.other} | {r.total} | {r.file_bytes} |\n"
            )
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def parse_binary_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--binary must be TARGET=/path/to/binary")
    name, path = value.split("=", maxsplit=1)
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--binary", action="append", type=parse_binary_arg, required=True)
    args = parser.parse_args()

    size_tool = find_size_tool()
    results = [measure(name, path, size_tool) for name, path in args.binary]
    results.sort(key=lambda r: (r.case, r.target))
    write_reports(results, args.result_dir, size_tool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
