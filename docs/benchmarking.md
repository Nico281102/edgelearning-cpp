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

For fair comparisons, use the same compiler family, optimization flags, topology, flat parameter layout, initial weights, synthetic samples, optimizer, batch policy, and iteration count.

For code-size comparisons, distinguish host linked-section size from firmware image size. The host report uses `llvm-size --format=sysv` and strips dead code with `-Os`, function/data sections, and linker dead stripping. MCU numbers should be regenerated with `arm-none-eabi-size` or the relevant vendor toolchain.
