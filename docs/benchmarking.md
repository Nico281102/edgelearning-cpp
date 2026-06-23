# Benchmarking

`benchmark_edgelearning_cpp` measures several Dense-only models:

- `Input<8>, Dense<16, ReLU>, Dense<1>`
- `Input<8>, Dense<32, ReLU>, Dense<16, ReLU>, Dense<1>`
- `Input<32>, Dense<64, ReLU>, Dense<32, ReLU>, Dense<4>`
- `Input<128>, Dense<64, ReLU>, Dense<32, ReLU>, Dense<3>`

It reports forward, backward, train step, optimizer step, zero-grad time, and memory breakdown.

`benchmark_regression_vs_c_baseline` writes `benchmarks/results/host_regression_report.md`. It records the baseline commit and the methodology for a local C comparison. The old C source must be checked out outside this repository and must not be vendored or published here.

For fair comparisons, use the same compiler family, optimization flags, topology, flat parameter layout, initial weights, synthetic samples, optimizer, batch policy, and iteration count.

