# Code Size Report

Baseline C commit: `0085814908ca1b57ece4fe367361d084fd74aa3e`.

The old C source is not vendored, copied, or published in this repository. When `EDGE_C_BASELINE_DIR` or `--c-baseline-dir` is provided, the script compiles a temporary harness outside this repository and records section sizes only.

## Methodology

- Topology: `1 -> Dense<4, Tanh> -> Dense<1, Linear>`
- Training path: one supervised sample step plus flush/update
- Optimization goal: code size, using `-Os`, function/data sections, and dead-code stripping
- Reported `total_section_bytes` is the sum of object sections from `llvm-size --format=sysv`; it is not the page-aligned executable file size.

## Results

| Implementation | Commit | Text | RoData | Data | BSS | Other | Total sections | File bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| edgelearning_cpp | `51f36c7` | 1644 | 16 | 24 | 4 | 0 | 1688 | 36136 |
| old_c_baseline | `0085814908ca1b57ece4fe367361d084fd74aa3e` | 6288 | 192 | 96 | 8 | 240 | 6824 | 36328 |

## Raw Sections

### edgelearning_cpp

- Compiler: `Apple clang version 17.0.0 (clang-1700.0.13.5)`
- Flags: `-Os -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -Wl,-dead_strip`
- Binary measured: `.tmp/build-propagate-main-release/benchmarks/code_size_cpp_regression`

```text
__text                   1620
__stubs                  24
__const                  16
__got                    24
__bss                    4
Total                    1688
```

### old_c_baseline

- Compiler: `Apple clang version 17.0.0 (clang-1700.0.13.5)`
- Flags: `-Os -ffunction-sections -fdata-sections -I ${EDGE_C_BASELINE_DIR} -I ${EDGE_C_BASELINE_DIR}/Inc -Wl,-dead_strip -lm`
- Binary measured: `<temporary build>/c_code_size_regression`

```text
__text                   6168
__stubs                  120
__const                  192
__unwind_info            240
__got                    96
__bss                    8
Total                    6824
```
