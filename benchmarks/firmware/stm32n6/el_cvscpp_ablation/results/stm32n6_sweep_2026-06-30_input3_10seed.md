# STM32N6 EL_C_vsCpp Per-Variant Sweep - 2026-06-30 - 10 seeds

Board target: STM32N6 Cortex-M55 with MVE.
Task: deterministic linear regression, input 3, output 1, batch 256.
Build/run unit: one firmware ELF per variant and per network size.
Protocol: Adam, rollout 1024, 2 epochs, 8 optimizer steps, 2048 sample-passes per measured run.
Batch semantics: all variants use mean-reduced minibatch gradients. C and EdgeLearning++ accumulate 256 per-sample gradients, scale by `1/256`, and then apply one Adam update; RLTools uses a static tensor with shape `[256, input_features]` and equivalent MSE mean reduction.
Warm-up: 2 full training runs per seed, with model and optimizer reset before the measured run.
Timing: pre-generated rollout hot path only; setup, import/export, reset, sample generation, warm-up, traces, and serial I/O are outside DWT.
Profiling: training-loop component counters are collected in a separate equivalent pass with the same initial parameters and dataset, then averaged over seeds.
Convergence trace: seed 0, minibatch MSE after each Adam update, emitted by an untimed diagnostic pass.
Build: static C++/RLTools model storage, all firmware objects compiled with `-Ofast`.
Network state is the comparable runtime-memory field: C uses arena plus control bytes, EdgeLearning++ uses the static model object, and RLTools uses the static runtime bundle needed to run the batch-256 network.
ELF size columns are from the same per-variant image used for the runtime row.

This report uses the public C++/RLTools variant set and does not require the legacy C checkout.

All per-variant runs completed with `DONE status=0`.

| Config | Input | Seeds | Warm-ups | Params | C++ M55 avg | C++ Generic avg | Generic/M55 | RLTools Generic avg | RLTools/M55 | RLTools/M55 runtime ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 3 | 10 | 2 | 113 | 2268742 | 2176513 | 0.959 | 3471548 | 1.530 | 1.530x |
| 16x8 | 3 | 10 | 2 | 209 | 4050408 | 4120807 | 1.017 | 6077509 | 1.500 | 1.500x |
| 16x16 | 3 | 10 | 2 | 353 | 5763641 | 6686004 | 1.160 | 11167784 | 1.938 | 1.938x |
| 32x16 | 3 | 10 | 2 | 673 | 10338492 | 12133369 | 1.174 | 30318127 | 2.933 | 2.933x |
| 32x32 | 3 | 10 | 2 | 1217 | 16851334 | 21773600 | 1.292 | 55399564 | 3.288 | 3.288x |
| 64x32 | 3 | 10 | 2 | 2369 | 27562567 | 40380652 | 1.465 | 108213756 | 3.926 | 3.926x |

| Config | M55 network state | Generic network state | RLTools network state | M55 ELF dec/file | Generic ELF dec/file | RLTools ELF dec/file |
|---|---:|---:|---:|---:|---:|---:|
| 8x8 | 2,080 | 1,976 | 58,192 | 72,584/72,692 | 75,088/74,788 | 133,912/79,752 |
| 16x8 | 3,680 | 3,576 | 92,496 | 74,744/72,672 | 76,248/73,716 | 169,768/80,112 |
| 16x16 | 6,048 | 5,944 | 111,184 | 77,952/72,192 | 81,840/75,660 | 191,416/82,476 |
| 32x16 | 11,296 | 11,192 | 181,840 | 85,448/72,044 | 90,184/76,316 | 261,884/79,208 |
| 32x32 | 20,128 | 20,024 | 223,312 | 98,920/72,160 | 102,976/75,820 | 306,076/77,980 |
| 64x32 | 38,816 | 38,712 | 372,816 | 127,888/73,388 | 132,928/77,980 | 466,500/79,240 |

Raw UART logs, `.size.txt` files, and ELF paths are referenced in the CSV.

<!-- plots:start -->
## Generated plots

![Speedup curve](stm32n6_speedup_2026-06-30_input3.svg)

![Training-loop component breakdown](stm32n6_training_component_breakdown_2026-06-30_input3.svg)

![Convergence trace](stm32n6_convergence_2026-06-30_input3_32x32.svg)

![Firmware ELF component breakdown](stm32n6_elf_component_breakdown_2026-06-30_input3.svg)
<!-- plots:end -->
