# Benchmarking

`benchmark_edgelearning_cpp` measures several Dense-only models:

- `Input<8>, Dense<16, ReLU>, Dense<1>`
- `Input<8>, Dense<32, ReLU>, Dense<16, ReLU>, Dense<1>`
- `Input<32>, Dense<64, ReLU>, Dense<32, ReLU>, Dense<4>`
- `Input<128>, Dense<64, ReLU>, Dense<32, ReLU>, Dense<3>`

It reports forward, backward, train step, optimizer step, zero-grad time, and memory breakdown.

`benchmark_mixed_precision` writes:

- `benchmarks/results/mixed_precision_summary.csv`
- `benchmarks/results/mixed_precision_convergence.csv`
- `benchmarks/results/mixed_precision_convergence.svg`
- `benchmarks/results/mixed_precision_report.md`

It compares the FP32 baseline with `edge::precision::MixedFP16` on a deterministic synthetic regression task. `MixedFP16` follows the common mixed-precision pattern of FP32 master parameters, gradients, accumulators, optimizer state, and loss values with FP16 activation storage when `_Float16` is available. The checked-in result is a MacBook M2 host measurement.

`benchmark_regression_vs_c_baseline` writes `benchmarks/results/host_regression_report.md`. It records the baseline commit and the methodology for a local C comparison. The old C source must be checked out outside this repository and must not be vendored or published here.

`benchmark_code_size` writes `benchmarks/results/code_size_report.md` and `.csv`. It measures a minimal regression-training binary for the C++ implementation. If `EDGE_C_BASELINE_DIR` points to a local old-C checkout at commit `0085814908ca1b57ece4fe367361d084fd74aa3e`, it also builds a temporary C harness outside this repository and reports the old-C linked section sizes.

The M55 regression sweep is built as one static binary per topology:

- `Input<32>, Dense<H1, ReLU>, Dense<H2, ReLU>, Dense<1, Linear>`
- hidden pairs: `8x8`, `16x8`, `16x16`, `32x16`, `32x32`, `32x64`, `64x64`, `128x64`
- batch size: 256
- task: deterministic synthetic linear regression

The always-on C++ targets are named `test_m55_regression_<H1>x<H2>` and compare
`Backend::M55` with `Backend::Generic`. Optional legacy-C targets are generated
only when configuring with `EDGE_C_BASELINE_DIR=/path/to/EdgeLearning`; the old
C source remains outside this repository.

The two legacy-C ablation families are:

- `test_legacy_c_<backend>_direct_backend_regression_<H1>x<H2>`: legacy C
  network versus a static C++ model whose Dense layers call the legacy C backend
  kernels directly.
- `test_legacy_c_<backend>_native_m55_regression_<H1>x<H2>`: legacy C network
  versus the static C++ model using `Backend::M55`.

Choose `-DEDGE_LEGACY_C_BACKEND=m55` for the old C M55 backend or `generic` for
host validation.

`m55_regression_elf_size` measures the generated static sweep ELFs and writes
`benchmarks/results/m55_regression_elf_size.csv` and `.md`. The `.bss` numbers
include the static C++ model storage or legacy C arena. Set `EDGE_SIZE_TOOL`
when a specific tool is required, for example `arm-none-eabi-size` in the
firmware toolchain.

## STM32N6 firmware C/C++/RLTools experiment

`firmware/el_cvscpp_ablation/` contains the on-target STM32N6 experiment. It is
separate from the host CMake benchmarks because it builds firmware images,
flashes the board, captures UART output, and measures training work with the
DWT cycle counter on the MCU.

The current firmware sweep uses:

- topology: `Input<3>, Dense<H1, ReLU>, Dense<H2, ReLU>, Dense<1, Linear>`
- hidden pairs: `8x8`, `16x8`, `16x16`, `32x16`, `32x32`, `64x32`
- variants: legacy C M55, C++ direct legacy-C backend, C++ native M55,
  C++ generic, RLTools generic
- storage: static C arena/control for C, static compile-time model storage for
  C++, and static RLTools runtime/model state for RLTools
- training: Adam, batch 256, 1024 rollout samples, 2 epochs, 8 optimizer
  steps, 2048 sample-passes, 10 deterministic seeds, and 2 warm-up runs per
  seed/variant
- build flags: `-Ofast`, function/data sections, and linker garbage collection

The timing window excludes setup, warm-up, convergence tracing, parameter
import/export, serial printing, and numerical comparisons. Only the hot
training work is measured: minibatch gradient reset, forward/backward sample
passes, and Adam updates. Firmware runs also emit component profiling counters
from a separate equivalent profiling pass with the same initial parameters and
dataset, so the primary speedup timings are not polluted by internal probes. C++
and RLTools variants split `zero_grad`, optional `input_copy`, `forward`, `loss`,
`backward`, and `adam_update`; legacy C reports its API-level
`sample_train_step` for the combined forward/loss/backward work.

The 2026-06-26 input-3 ten-seed run is consistent with the expected RLTools
behavior for small static networks on this firmware path: RLTools is faster than
legacy C on `8x8` and `16x8`, then slower from `16x16` upward. The C++ direct
legacy-C-backend variant stays close to the legacy C baseline on the smaller
and mid-size cases, then exposes adapter/layout overhead on the larger cases.
The native C++ M55 backend is faster than legacy C across the measured sweep.

Cycle ratios below are variant cycles divided by legacy C cycles; values below
`1.0` are faster than legacy C.

| Hidden | Direct C backend | C++ M55 | C++ Generic | RLTools Generic |
|---|---:|---:|---:|---:|
| `8x8` | 0.950 | 0.457 | 0.440 | 0.621 |
| `16x8` | 0.976 | 0.631 | 0.645 | 0.855 |
| `16x16` | 0.991 | 0.705 | 0.823 | 1.194 |
| `32x16` | 0.984 | 0.859 | 1.008 | 2.077 |
| `32x32` | 1.230 | 0.975 | 1.256 | 2.953 |
| `64x32` | 1.183 | 0.877 | 1.328 | 3.474 |

Model footprint is measured separately from runtime. For C, the model-state
number is the static arena plus control state. For C++ variants, the report
records the compile-time required memory and static model object size. For
RLTools, it records the static runtime state plus model object. Firmware image
size is measured from one separate ELF per variant and topology with
`arm-none-eabi-size`; each ELF still includes the common benchmark harness and
static rollout buffers.

| Variant | Model-state bytes across sweep | ELF `dec` bytes across sweep |
|---|---:|---:|
| Legacy C M55 | 3,296-40,160 | 80,884-135,820 |
| C++ direct legacy-C backend | 2,080-38,944 | 78,140-132,988 |
| C++ native M55 | 2,080-38,944 | 72,492-127,916 |
| C++ generic | 1,976-38,840 | 74,996-132,964 |
| RLTools generic | 1,948-38,684 | 76,868-134,292 |

Generated artifacts for this run are under
`firmware/el_cvscpp_ablation/results/`, including the ten-seed sweep table, the
per-variant ELF/model-size table, the speedup/convergence CSV/SVG files, the
training-loop component-breakdown CSV/SVG, and the ELF component-breakdown
CSV/SVG. The ELF component-breakdown plot uses the firmware ELF `text`, `data`,
and `bss` sections. The legacy C source and RLTools source are referenced
through local paths and are not vendored into this repository.

For fair comparisons, use the same compiler family, optimization flags, topology, flat parameter layout, initial weights, synthetic samples, optimizer, batch policy, and iteration count.

For code-size comparisons, distinguish host linked-section size from firmware image size. The host report uses `llvm-size --format=sysv` and strips dead code with `-Os`, function/data sections, and linker dead stripping. MCU numbers should be regenerated with `arm-none-eabi-size` or the relevant vendor toolchain.
