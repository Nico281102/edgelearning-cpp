# EdgeLearning++ Benchmarks

`benchmark_edgelearning_cpp` measures the C++20 implementation directly across several Dense-only network sizes. It reports per-call nanoseconds for forward, backward, `train_step`, optimizer step, and `zero_grad`, plus the compile-time memory breakdown.

`benchmark_regression_vs_c_baseline` generates `benchmarks/results/host_regression_report.md`. It does not include the old C implementation. To compare against the C baseline, check out `Nico281102/EdgeLearning` at commit `0085814908ca1b57ece4fe367361d084fd74aa3e` outside this repository and run the same topology, seed, synthetic samples, optimizer, batch policy, compiler, and flags.

If the old C benchmark output exists at `/tmp/edgelearning_c_baseline_008581_benchmark.txt`, or if `EDGE_C_BASELINE_BENCHMARK_LOG` points to another local output file, the regression report includes selected measurement blocks. The log itself is not committed.

`benchmark_code_size` builds the C++ minimal regression-training target and writes `benchmarks/results/code_size_report.md` plus `.csv`. If `EDGE_C_BASELINE_DIR` points to a local old-C checkout at the baseline commit, it also builds a temporary C harness outside this repository and records the old-C section sizes. The old C source and temporary harness are not committed.

`benchmark_mixed_precision` compares FP32 against `edge::precision::MixedFP16` on a deterministic synthetic regression task and writes `results/mixed_precision_report.md`, `results/mixed_precision_summary.csv`, `results/mixed_precision_convergence.csv`, and `results/mixed_precision_convergence.svg`. The checked-in measurement is a MacBook M2 host result.

`test_m55_regression_<hidden1>x<hidden2>` builds one static-model ELF per M55
regression-sweep topology:

- input features: 32
- hidden layers: `8x8`, `16x8`, `16x16`, `32x16`, `32x32`, `32x64`, `64x64`, `128x64`
- output: 1 linear neuron
- batch: 256 samples, accumulated before one SGD update

Each C++ test compares `Backend::M55` against `Backend::Generic` with identical
parameters and synthetic linear-regression samples. If `EDGE_C_BASELINE_DIR`
points to a local legacy C checkout, CMake also builds two ablation families
that link the old C files from that external path without copying them into this
repository:

- `test_legacy_c_<backend>_direct_backend_regression_<hidden1>x<hidden2>`:
  legacy C network versus a static C++ model whose Dense layers call the legacy
  C backend kernels directly.
- `test_legacy_c_<backend>_native_m55_regression_<hidden1>x<hidden2>`:
  legacy C network versus the static C++ model using `Backend::M55`.

Use `-DEDGE_LEGACY_C_BACKEND=m55` for the firmware/Cortex-M55 comparison, plus
`-DEDGE_LEGACY_C_INCLUDE_DIRS=...` if the toolchain needs CMSIS include paths.

`m55_regression_elf_size` measures the generated static sweep ELFs and writes
`results/m55_regression_elf_size.csv` and `.md`. Set `EDGE_SIZE_TOOL` to force a
specific size tool, for example `arm-none-eabi-size`.

The on-target STM32N6 firmware experiment, including C, C++ M55, C++ generic,
C++ direct legacy-C backend, and RLTools generic variants, lives under
`firmware/el_cvscpp_ablation/`. See `docs/benchmarking.md` for the firmware
methodology, cycle-ratio summary, model-state measurement, and per-variant ELF
size notes.

Example:

```sh
cmake -S . -B build -DEDGE_BUILD_BENCHMARKS=ON
cmake --build build --target code_size_cpp_regression
cmake --build build --target benchmark_mixed_precision
./build/benchmarks/benchmark_mixed_precision
EDGE_C_BASELINE_DIR=/path/to/EdgeLearning-at-0085814908ca1b57ece4fe367361d084fd74aa3e \
cmake --build build --target benchmark_code_size
cmake --build build --target m55_regression_elf_size
```

The current host measurement for topology `1 -> Dense<4,Tanh> -> Dense<1>` is recorded in `results/code_size_report.md`. Treat it as a host linked-section comparison. For MCU flash/SRAM size, rebuild both implementations with the embedded toolchain and run the target `size` tool.
