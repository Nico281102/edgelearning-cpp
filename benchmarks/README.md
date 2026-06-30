# EdgeLearning++ Benchmarks

Benchmark code and generated artifacts are grouped by platform and experiment.

| Path | Purpose |
|---|---|
| `host/m2/runtime/` | Dense-only host runtime microbenchmarks on MacBook M2 |
| `host/m2/batch_sweep/` | Host batch-size sweep results |
| `host/m2/mixed_precision/` | Host FP32 vs mixed-precision convergence smoke test |
| `host/m2/c_baseline_regression/` | Host C++ vs optional private C baseline methodology |
| `firmware/stm32n6/m55_regression/` | Static ELF-size sweep for M55 regression binaries |
| `firmware/stm32n6/el_cvscpp_ablation/` | On-target STM32N6 runtime, model-size, and ELF reports |
| `mixed/` | Reserved for campaigns that pair host and firmware measurements |

`benchmark_edgelearning_cpp` measures the C++20 implementation directly across
several Dense-only network sizes. It reports per-call nanoseconds for forward,
backward, `train_step`, optimizer step, and `zero_grad`, plus the compile-time
memory breakdown.

`benchmark_mixed_precision` compares FP32 against
`edge::precision::MixedFP16` on a deterministic synthetic regression task and
writes results under `host/m2/mixed_precision/results/`.

`benchmark_regression_vs_c_baseline` writes
`host/m2/c_baseline_regression/results/host_regression_report.md`. It does not
include the old C implementation. To compare against the C baseline, check out
`Nico281102/EdgeLearning` at commit
`0085814908ca1b57ece4fe367361d084fd74aa3e` outside this repository and run the
same topology, seed, synthetic samples, optimizer, batch policy, compiler, and
flags.

`benchmark_code_size` builds the C++ minimal regression-training target and
writes `host/m2/c_baseline_regression/results/code_size_report.md` plus `.csv`.
If `EDGE_C_BASELINE_DIR` points to a local old-C checkout at the baseline
commit, it also builds a temporary C harness outside this repository and records
the old-C section sizes. The old C source and temporary harness are not
committed.

`m55_regression_elf_size` measures generated static sweep ELFs and writes
`firmware/stm32n6/m55_regression/results/m55_regression_elf_size.csv` and
`.md`. Set `EDGE_SIZE_TOOL` to force a specific size tool, for example
`arm-none-eabi-size`.

The on-target STM32N6 firmware launcher lives under
`firmware/el_cvscpp_ablation/`. Its public sweep compares C++ M55, C++ generic,
and RLTools generic with static batch-256 tensors. Legacy C and C++ direct
legacy-C backend variants require a private external C checkout. The generated
reports live under
`benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/`.

Example:

```sh
cmake -S . -B build -DEDGE_BUILD_BENCHMARKS=ON
cmake --build build --target benchmark_mixed_precision
./build/benchmarks/benchmark_mixed_precision
cmake --build build --target benchmark_code_size
cmake --build build --target m55_regression_elf_size
```

See [../docs/benchmarking/README.md](../docs/benchmarking/README.md) for the
measurement protocol and interpretation notes.
