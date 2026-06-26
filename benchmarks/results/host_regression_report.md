# Host Regression Report

Baseline C commit: `0085814908ca1b57ece4fe367361d084fd74aa3e`.

C++ commit used: `5690c2e (dirty)`.

This repository does not vendor or republish the old C source. The old C baseline is intended to be cloned or checked out locally outside `edgelearning-cpp` for side-by-side measurement using the same topology, seed, synthetic dataset, optimizer, and batch policy.

## Host And Build

- Compiler: AppleClang 17.0.0.17000013
- Language mode: C++20
- Core flags: `-fno-exceptions -fno-rtti`
- Host CPU/model: Apple M2
- Hardware threads: 8

## Current C++ Measurement

- Topology: `InputVector<8>, Dense<32, ReLU>, Dense<16, ReLU>, Dense<1>`
- Iterations: 1000
- Optimizer: Adam, learning rate 0.01
- Batch reduction: Mean, batch size 8
- C++ train_step mean: 13595.5 ns
- Parameter bytes: 3332
- Gradient bytes: 3332
- Optimizer bytes: 6664
- Activation bytes: 228
- Workspace bytes: 192
- Total planned bytes: 13748

## Code Size

Code-size measurements are available in `code_size_report.md`. That report compares the linked C++ minimal training binary against the old C baseline when `EDGE_C_BASELINE_DIR` was provided.

## Old C Baseline Local Measurements

Source checkout location used for this run: temporary directory outside this repository.

Raw baseline benchmark log: `/tmp/edgelearning_c_baseline_008581_benchmark.txt` (not committed to this repository).

Selected result blocks:

```text
=== DATASET: simple_regression ===
samples=64 epochs=300 input_dim=1 loss=mse optimizer=sgd
network: 1 -> 2(relu) -> 1(none)
batches: {1, 8, 16, 32, 64}
trend baseline: first row of this dataset block
PERFORMANCE TABLE
+-------+------------+------------+---------+---------+----------------+---------+---------+---------+---------+---------+--------+----------------+
| batch | lr_eff     | loss       | fit_ms  | step_us | prof fwd/bwd/upd| e1      | e2      | e3      | e4      | e5      | d_fit%| d_prof f/b/u   |
+-------+------------+------------+---------+---------+----------------+---------+---------+---------+---------+---------+--------+----------------+
|     1 |   0.050000 |   0.000752 |   1.063 |   0.055 | 0/0/0          | 0.20535 | 0.18733 | 0.16932 | 0.15131 | 0.13330 |    0.0 | baseline       |
|     8 |   0.006250 |   0.000000 |   0.633 |   0.033 | 0/0/0          | 0.00052 | 0.00002 | 0.00002 | 0.00002 | 0.00002 |  -40.5 | +0.0/+0.0/+0.0 |
|    16 |   0.003125 |   0.000002 |   0.597 |   0.031 | 0/0/0          | 0.00851 | 0.00052 | 0.00051 | 0.00049 | 0.00048 |  -43.8 | +0.0/+0.0/+0.0 |
|    32 |   0.001563 |   0.000042 |   0.579 |   0.030 | 0/0/0          | 0.03064 | 0.00994 | 0.00280 | 0.00272 | 0.00264 |  -45.5 | +0.0/+0.0/+0.0 |
|    64 |   0.000781 |   0.000449 |   0.543 |   0.028 | 0/0/0          | 0.06905 | 0.04738 | 0.02571 | 0.00403 | 0.01354 |  -48.9 | +0.0/+0.0/+0.0 |
+-------+------------+------------+---------+---------+----------------+---------+---------+---------+---------+---------+--------+----------------+

PROFILE TABLE (cycles)
+-------+--------+--------+--------+--------+--------+-----------+-----------+-----------+--------+--------+
| batch | fwd    | loss   | bwd    | zero   | upd    | fit_c/ep  | fit_c/in  | est_c/in  | d_fit% | d_est% |
+-------+--------+--------+--------+--------+--------+-----------+-----------+-----------+--------+--------+
|     1 |      0 |      0 |      0 |      0 |      0 |       0.0 |     0.000 |     0.000 |    0.0 |    0.0 |
|     8 |      0 |      0 |      0 |      0 |      0 |       0.0 |     0.000 |     0.000 |    0.0 |    0.0 |
|    16 |      0 |      0 |      0 |      0 |      0 |       0.0 |     0.000 |     0.000 |    0.0 |    0.0 |
|    32 |      0 |      0 |      0 |      0 |      0 |       0.0 |     0.000 |     0.000 |    0.0 |    0.0 |
|    64 |      0 |      0 |      0 |      0 |      0 |       0.0 |     0.000 |     0.000 |    0.0 |    0.0 |
+-------+--------+--------+--------+--------+--------+-----------+-----------+-----------+--------+--------+

ARENA / WORKSPACE TABLE
+-------+----------+----------+---------+----------+----------+-----------+-----------+
| batch | in_bytes | usable   | trimmed | used     | free     | used%     | ws req/cap|
+-------+----------+----------+---------+----------+----------+-----------+-----------+
|     1 |   131072 |   131072 |       0 |      848 |   130224 |    0.65% |    2/2     |
|     8 |   131072 |   131072 |       0 |      848 |   130224 |    0.65% |    2/2     |
|    16 |   131072 |   131072 |       0 |      848 |   130224 |    0.65% |    2/2     |
|    32 |   131072 |   131072 |       0 |      848 |   130224 |    0.65% |    2/2     |
|    64 |   131072 |   131072 |       0 |      848 |   130224 |    0.65% |    2/2     |
+-------+----------+----------+---------+----------+----------+-----------+-----------+

=== DATASET: multi_regression ===
samples=128 epochs=1500 input_dim=8 loss=mse optimizer=adam
network: 8 -> 32(relu) -> 16(relu) -> 1(none)
batches: {1, 8, 16, 32, 64}
trend baseline: first row of this dataset block
PERFORMANCE TABLE
+-------+------------+------------+---------+---------+----------------+---------+---------+---------+---------+---------+--------+----------------+
| batch | lr_eff     | loss       | fit_ms  | step_us | prof fwd/bwd/upd| e1      | e2      | e3      | e4      | e5      | d_fit%| d_prof f/b/u   |
+-------+------------+------------+---------+---------+----------------+---------+---------+---------+---------+---------+--------+----------------+
|     1 |   0.010000 |   0.005702 | 106.065 |   0.552 | 0/0/0          | 0.03345 | 0.14088 | 0.04834 | 0.00243 | 0.00984 |    0.0 | baseline       |
|     8 |   0.001250 |   0.025490 |  74.263 |   0.387 | 0/0/0          | 0.05402 | 0.14291 | 0.04876 | 0.07088 | 0.22580 |  -30.0 | +0.0/+0.0/+0.0 |
|    16 |   0.000625 |   0.027667 |  67.711 |   0.353 | 0/0/0          | 0.03214 | 0.15106 | 0.06241 | 0.09140 | 0.32166 |  -36.2 | +0.0/+0.0/+0.0 |
|    32 |   0.000312 |   0.002016 |  69.783 |   0.363 | 0/0/0          | 0.01696 | 0.04723 | 0.03889 | 0.03696 | 0.00601 |  -34.2 | +0.0/+0.0/+0.0 |
|    64 |   0.000156 |   0.016134 |  67.746 |   0.353 | 0/0/0          | 0.03748 | 0.12775 | 0.03837 | 0.02810 | 0.23498 |  -36.1 | +0.0/+0.0/+0.0 |
+-------+------------+------------+---------+---------+----------------+---------+---------+---------+---------+---------+--------+----------------+

PROFILE TABLE (cycles)
+-------+--------+--------+--------+--------+--------+-----------+-----------+-----------+--------+--------+
| batch | fwd    | loss   | bwd    | zero   | upd    | fit_c/ep  | fit_c/in  | est_c/in  | d_fit% | d_est% |
+-------+--------+--------+--------+--------+--------+-----------+-----------+-----------+--------+--------+
|     1 |      0 |      0 |      0 |      0 |      0 |       0.0 |     0.000 |     0.000 |    0.0 |    0.0 |
|     8 |      0 |      0 |      0 |      0 |      0 |       0.0 |     0.000 |     0.000 |    0.0 |    0.0 |
|    16 |      0 |      0 |      0 |      0 |      0 |       0.0 |     0.000 |     0.000 |    0.0 |    0.0 |
|    32 |      0 |      0 |      0 |      0 |      0 |       0.0 |     0.000 |     0.000 |    0.0 |    0.0 |
|    64 |      0 |      0 |      0 |      0 |      0 |       0.0 |     0.000 |     0.000 |    0.0 |    0.0 |
+-------+--------+--------+--------+--------+--------+-----------+-----------+-----------+--------+--------+

ARENA / WORKSPACE TABLE
+-------+----------+----------+---------+----------+----------+-----------+-----------+
| batch | in_bytes | usable   | trimmed | used     | free     | used%     | ws req/cap|
+-------+----------+----------+---------+----------+----------+-----------+-----------+
|     1 |   131072 |   131072 |       0 |    15072 |   116000 |   11.50% |   32/32    |
|     8 |   131072 |   131072 |       0 |    15072 |   116000 |   11.50% |   32/32    |
|    16 |   131072 |   131072 |       0 |    15072 |   116000 |   11.50% |   32/32    |
|    32 |   131072 |   131072 |       0 |    15072 |   116000 |   11.50% |   32/32    |
|    64 |   131072 |   131072 |       0 |    15072 |   116000 |   11.50% |   32/32    |
+-------+----------+----------+---------+----------+----------+-----------+-----------+

```

## C Baseline Methodology

1. Check out `Nico281102/EdgeLearning` at the baseline commit outside this repository.
2. Build the old C library with the same compiler family and optimization flags.
3. Use the same flat parameter layout: layer order, row-major weights, then bias.
4. Run the same synthetic samples for the same iteration count and report mean timing.
5. Publish only measurements and methodology, not the old C source.
