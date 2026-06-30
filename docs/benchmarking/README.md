# Benchmarking

`benchmark_edgelearning_cpp` measures several Dense-only models:

- `InputVector<8>, Dense<16, ReLU>, Dense<1>`
- `InputVector<8>, Dense<32, ReLU>, Dense<16, ReLU>, Dense<1>`
- `InputVector<32>, Dense<64, ReLU>, Dense<32, ReLU>, Dense<4>`
- `InputVector<128>, Dense<64, ReLU>, Dense<32, ReLU>, Dense<3>`

It reports forward, backward, train step, optimizer step, zero-grad time, and memory breakdown.

`benchmark_mixed_precision` writes:

- `benchmarks/host/m2/mixed_precision/results/mixed_precision_summary.csv`
- `benchmarks/host/m2/mixed_precision/results/mixed_precision_convergence.csv`
- `benchmarks/host/m2/mixed_precision/results/mixed_precision_convergence.svg`
- `benchmarks/host/m2/mixed_precision/results/mixed_precision_report.md`

It compares the FP32 baseline with `edge::precision::MixedFP16` on a deterministic synthetic regression task. `MixedFP16` follows the common mixed-precision pattern of FP32 master parameters, gradients, accumulators, optimizer state, and loss values with FP16 activation storage when `_Float16` is available. The checked-in result is a MacBook M2 host measurement.

`benchmark_regression_vs_c_baseline` writes `benchmarks/host/m2/c_baseline_regression/results/host_regression_report.md`. It records the baseline commit and the methodology for a local C comparison. The old C source must be checked out outside this repository and must not be vendored or published here.

`benchmark_code_size` writes `benchmarks/host/m2/c_baseline_regression/results/code_size_report.md` and `.csv`. It measures a minimal regression-training binary for the C++ implementation. If `EDGE_C_BASELINE_DIR` points to a local old-C checkout at commit `0085814908ca1b57ece4fe367361d084fd74aa3e`, it also builds a temporary C harness outside this repository and reports the old-C linked section sizes.

The M55 regression sweep is built as one static binary per topology:

- `InputVector<32>, Dense<H1, ReLU>, Dense<H2, ReLU>, Dense<1, Linear>`
- hidden pairs: `8x8`, `16x8`, `16x16`, `32x16`, `32x32`, `64x32`
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
`benchmarks/firmware/stm32n6/m55_regression/results/m55_regression_elf_size.csv` and `.md`. The `.bss` numbers
include the static C++ model storage or legacy C arena. Set `EDGE_SIZE_TOOL`
when a specific tool is required, for example `arm-none-eabi-size` in the
firmware toolchain.

## STM32N6 firmware C/C++/RLTools experiment

`firmware/el_cvscpp_ablation/` contains the on-target STM32N6 experiment. It is
separate from the host CMake benchmarks because it builds firmware images,
flashes the board, captures UART output, and measures training work with the
DWT cycle counter on the MCU. The public firmware path compares EdgeLearning++
M55, EdgeLearning++ generic, and RLTools generic. Legacy-C rows require a
private external C checkout and are not fully reproducible from this repository
alone.

The current firmware sweep uses:

- topology: `InputVector<3>, Dense<H1, ReLU>, Dense<H2, ReLU>, Dense<1, Linear>`
- hidden pairs: `8x8`, `16x8`, `16x16`, `32x16`, `32x32`, `64x32`
- public variants: C++ native M55, C++ generic, RLTools generic with static
  batch-256 tensors
- private legacy-C variants: legacy C M55 and C++ direct legacy-C backend
- storage: static C arena/control for C, static compile-time model storage for
  C++, and static RLTools runtime/model state for RLTools
- training: Adam, batch 256, 1024 rollout samples, 2 epochs, 8 optimizer
  steps, 2048 sample-passes, 10 deterministic seeds, and 2 warm-up runs per
  seed/variant
- build flags: `-Ofast`, function/data sections, and linker garbage collection

The timing window excludes setup, warm-up, convergence tracing, parameter
import/export, serial printing, and numerical comparisons. Only the hot
training work is measured: minibatch gradient reset, forward/backward work, and
Adam updates. EdgeLearning++ performs one forward/loss/backward pass per sample
and applies Adam after accumulating 256 gradients. RLTools uses one static
`[256, input_features]` input tensor and one forward/loss/backward/update per
minibatch. Firmware runs also emit component profiling counters from a separate
equivalent profiling pass with the same initial parameters and dataset, so the
primary speedup timings are not polluted by internal probes. C++ and RLTools
variants split `zero_grad`, optional `input_copy`, `forward`, `loss`,
`backward`, and `adam_update`; legacy C reports its API-level
`sample_train_step` for the combined forward/loss/backward work.

Model footprint is measured separately from runtime. For C, the model-state
number is the static arena plus control state. For C++ variants, the report
records the compile-time required memory and static model object size. For
RLTools, it records the static runtime state plus model object. Firmware image
size is measured from one separate ELF per variant and topology with
`arm-none-eabi-size`; each ELF still includes the common benchmark harness and
static rollout buffers.

The checked-in generated report is the source of record for public firmware
numbers:
`benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/stm32n6_sweep_2026-06-30_input3_10seed.md`.
Users can regenerate the C++/RLTools rows by running the default public sweep.
Private legacy-C rows can be regenerated only with an external C checkout. The
accompanying CSV contains raw cycle averages, model-state fields, ELF paths,
and status columns. Plot CSV/SVG files in the same directory provide speedup,
convergence, training-loop component breakdown, and ELF section breakdown.

README.md includes a short RLTools-baseline preview. Avoid copying the full
firmware tables into multiple documents; regenerate or link the report instead.

### Firmware Variant Interpretation

The STM32N6 report contains the rows needed for the main four-way comparison:

| Variant | Meaning |
|---|---|
| `legacy_c` | `EL-C M55`, the private legacy C runtime with the M55 backend |
| `cpp_generic` | `EL++ generic scalar`, the public C++ model without specialized M55 kernels |
| `cpp_m55` | `EL++ M55`, the same public C++ model using the M55 backend policy |
| `rltools_generic` | `RLTools generic/static batch`, the external C++ baseline with static batch-256 tensors |

The `cpp_direct_c_backend` row is an additional ablation: it keeps the C++ model
layout while calling the legacy C backend kernels directly. It is useful for
debugging backend effects, but it is not required for the four-way public
presentation.

For runtime ratios, use `cycles_avg` from the CSV. For model size, use
`legacy_c_arena_bytes + legacy_c_control_bytes` for `EL-C M55`,
`cpp_*_model_object` for owning C++ models, and `rltools_static_state` for the
RLTools runtime bundle. For deployable image footprint, use the per-variant
`*_elf_text`, `*_elf_data`, `*_elf_bss`, `*_elf_dec`, and `*_elf_file_bytes`
columns.

For fair comparisons, use the same compiler family, optimization flags, topology, flat parameter layout, initial weights, synthetic samples, optimizer, batch policy, and iteration count.

For code-size comparisons, distinguish host linked-section size from firmware image size. The host report uses `llvm-size --format=sysv` and strips dead code with `-Os`, function/data sections, and linker dead stripping. MCU numbers should be regenerated with `arm-none-eabi-size` or the relevant vendor toolchain.
