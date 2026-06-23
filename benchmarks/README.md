# EdgeLearning++ Benchmarks

`benchmark_edgelearning_cpp` measures the C++20 implementation directly across several Dense-only network sizes. It reports per-call nanoseconds for forward, backward, `train_step`, optimizer step, and `zero_grad`, plus the compile-time memory breakdown.

`benchmark_regression_vs_c_baseline` generates `benchmarks/results/host_regression_report.md`. It does not include the old C implementation. To compare against the C baseline, check out `Nico281102/EdgeLearning` at commit `0085814908ca1b57ece4fe367361d084fd74aa3e` outside this repository and run the same topology, seed, synthetic samples, optimizer, batch policy, compiler, and flags.

If the old C benchmark output exists at `/tmp/edgelearning_c_baseline_008581_benchmark.txt`, or if `EDGE_C_BASELINE_BENCHMARK_LOG` points to another local output file, the regression report includes selected measurement blocks. The log itself is not committed.

`benchmark_code_size` builds the C++ minimal regression-training target and writes `benchmarks/results/code_size_report.md` plus `.csv`. If `EDGE_C_BASELINE_DIR` points to a local old-C checkout at the baseline commit, it also builds a temporary C harness outside this repository and records the old-C section sizes. The old C source and temporary harness are not committed.

Example:

```sh
cmake -S . -B build -DEDGE_BUILD_BENCHMARKS=ON
cmake --build build --target code_size_cpp_regression
EDGE_C_BASELINE_DIR=/path/to/EdgeLearning-at-0085814908ca1b57ece4fe367361d084fd74aa3e \
cmake --build build --target benchmark_code_size
```

The current host measurement for topology `1 -> Dense<4,Tanh> -> Dense<1>` is recorded in `results/code_size_report.md`. Treat it as a host linked-section comparison. For MCU flash/SRAM size, rebuild both implementations with the embedded toolchain and run the target `size` tool.
