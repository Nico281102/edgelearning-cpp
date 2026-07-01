# STM32N6 EL_C_vsCpp Per-Variant Sweep - 2026-06-30 - 10 seeds

Board target: STM32N6 Cortex-M55 with MVE.
Task: deterministic linear regression, input 3, output 1, batch 256.
Build/run unit: one firmware ELF per variant and per network size.
Protocol: Adam, rollout 1024, 2 epochs, 8 optimizer steps, 2048 sample-passes per measured run.
Batch semantics: all variants use mean-reduced minibatch gradients. C and EdgeLearning++ accumulate 256 per-sample gradients, scale by `1/256`, and then apply one Adam update; RLTools uses a static tensor with shape `[256, input_features]` and equivalent MSE mean reduction.
RLTools topology: generic/static RLTools layer path aligned with the fast RL firmware network selection. Symmetric hidden sizes use `standardize -> MLP`; asymmetric hidden sizes use `standardize -> dense(H1) -> MLP tail(H2, 1)`. Standardization is initialized as identity.
Warm-up: 2 full training runs per seed, with model and optimizer reset before the measured run.
Timing: pre-generated rollout hot path only; setup, import/export, reset, sample generation, warm-up, traces, and serial I/O are outside DWT.
Profiling: training-loop component counters are collected in a separate equivalent pass with the same initial parameters and dataset, then averaged over seeds.
Legacy C exposes `sample_train_step` as one combined forward/loss/backward component because those operations are encapsulated by the C API.
Convergence trace: seed 0, minibatch MSE after each Adam update, emitted by an untimed diagnostic pass.
Build: static C arena and static C++ model, all firmware objects compiled with `-Ofast`.
Network state is the comparable runtime-memory field: C uses arena plus control bytes, EdgeLearning++ uses the static model object, and RLTools uses the static runtime bundle needed to run the batch-256 network.
ELF size columns are from the same per-variant image used for the runtime row.

This report includes private legacy-C ablation rows from an external checkout.

All per-variant runs completed with `DONE status=0`.

| Config | Input | Seeds | Warm-ups | Params | C M55 avg | Direct C-backend avg | Direct/C | C++ M55 avg | M55/C | C++ Generic avg | Generic/C | RLTools Generic avg | RLTools/C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 3 | 10 | 2 | 113 | 4992796 | 4731928 | 0.948 | 2268751 | 0.454 | 2176513 | 0.436 | 3568002 | 0.715 |
| 16x8 | 3 | 10 | 2 | 209 | 6435488 | 6209393 | 0.965 | 4050417 | 0.629 | 4120821 | 0.640 | 6683693 | 1.039 |
| 16x16 | 3 | 10 | 2 | 353 | 8220780 | 8063582 | 0.981 | 5764953 | 0.701 | 6685968 | 0.813 | 12541769 | 1.526 |
| 32x16 | 3 | 10 | 2 | 673 | 11819103 | 11789606 | 0.998 | 10338742 | 0.875 | 12128226 | 1.026 | 24156038 | 2.044 |
| 32x32 | 3 | 10 | 2 | 1217 | 18418792 | 17540907 | 0.952 | 16851824 | 0.915 | 21773553 | 1.182 | 47228137 | 2.564 |
| 64x32 | 3 | 10 | 2 | 2369 | 32062087 | 35910609 | 1.120 | 27658687 | 0.863 | 40688506 | 1.269 | 87535977 | 2.730 |

| Config | C network state | Direct network state | M55 network state | Generic network state | RLTools network state | C ELF dec/file | Direct ELF dec/file | M55 ELF dec/file | Generic ELF dec/file | RLTools ELF dec/file |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 3,296 | 2,080 | 2,080 | 1,976 | 61,284 | 81,332/80,628 (1.000x) | 78,668/78,732 (0.967x) | 72,584/72,692 (0.892x) | 75,088/74,788 (0.923x) | 137,248/78,396 (1.688x) |
| 16x8 | 4,928 | 3,680 | 3,680 | 3,576 | 87,396 | 83,764/80,692 (1.000x) | 81,108/78,924 (0.968x) | 74,744/72,672 (0.892x) | 76,248/73,716 (0.910x) | 165,128/80,136 (1.971x) |
| 16x16 | 7,264 | 6,048 | 6,048 | 5,944 | 114,276 | 87,244/80,660 (1.000x) | 84,588/78,772 (0.970x) | 77,952/72,192 (0.893x) | 81,840/75,660 (0.938x) | 194,384/80,752 (2.228x) |
| 32x16 | 12,576 | 11,296 | 11,296 | 11,192 | 168,548 | 95,132/80,660 (1.000x) | 92,468/78,964 (0.972x) | 85,448/72,044 (0.898x) | 90,184/76,316 (0.948x) | 250,380/80,572 (2.632x) |
| 32x32 | 21,344 | 20,128 | 20,128 | 20,024 | 226,404 | 108,244/80,660 (1.000x) | 105,620/78,804 (0.976x) | 98,920/72,160 (0.914x) | 102,976/75,820 (0.951x) | 310,764/77,952 (2.871x) |
| 64x32 | 40,160 | 38,816 | 38,816 | 38,712 | 343,140 | 136,300/80,724 (1.000x) | 133,532/78,884 (0.980x) | 127,888/73,388 (0.938x) | 132,928/77,980 (0.975x) | 439,252/81,228 (3.223x) |

Raw UART logs, `.size.txt` files, and ELF paths are referenced in the CSV.

<!-- plots:start -->
## Generated plots

![Speedup curve](stm32n6_speedup_2026-06-30_input3.svg)

![Training-loop component breakdown](stm32n6_training_component_breakdown_2026-06-30_input3.svg)

![Convergence trace](stm32n6_convergence_2026-06-30_input3_32x32.svg)

![Firmware ELF component breakdown](stm32n6_elf_component_breakdown_2026-06-30_input3.svg)
<!-- plots:end -->
