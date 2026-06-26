# STM32N6 EL_C_vsCpp Per-Variant Sweep - 2026-06-26

Build: one firmware ELF per variant and per network size, all with `-Ofast`.
Network input features: `3`.
Runtime: Adam, 10 seeds, 2 warm-up runs per seed/variant, batch 256, 1024 rollout samples, 2 epochs, 8 optimizer steps.
Timing excludes setup, warm-up, convergence tracing, export, and numerical comparisons.
Runtime columns use per-variant logs when available, otherwise the combined `variant=all` sweep.
ELF columns report `arm-none-eabi-size dec` and on-disk ELF bytes.
Each ELF still includes the common benchmark harness and static rollout buffers.
All runtime sources completed with `status=0`: `1`.
Model-size metadata fallback: `/Users/domenicososta/projects/edgelearning-cpp/firmware/el_cvscpp_ablation/results/stm32n6_sweep_2026-06-26_input3_10seed.csv`.

| Config | Input | C cycles | Direct C-backend cycles | Direct/C | C++ M55 cycles | M55/C | C++ Generic cycles | Generic/C | RLTools Generic cycles | RLTools/C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 3 | 5,016,850 | 4,750,684 | 0.947 | 2,287,993 | 0.456 | 2,213,466 | 0.441 | 3,093,011 | 0.617 |
| 16x8 | 3 | 6,471,087 | 6,312,141 | 0.975 | 4,082,979 | 0.631 | 4,169,624 | 0.644 | 5,530,348 | 0.855 |
| 16x16 | 3 | 8,240,886 | 8,168,195 | 0.991 | 5,806,241 | 0.705 | 6,779,423 | 0.823 | 9,844,205 | 1.195 |
| 32x16 | 3 | 11,803,468 | 11,859,089 | 1.005 | 10,366,813 | 0.878 | 12,175,991 | 1.032 | 25,041,861 | 2.122 |
| 32x32 | 3 | 17,280,759 | 21,242,599 | 1.229 | 16,850,997 | 0.975 | 21,755,360 | 1.259 | 51,007,076 | 2.952 |
| 64x32 | 3 | 33,037,271 | 34,738,886 | 1.052 | 27,555,437 | 0.834 | 41,032,636 | 1.242 | 107,631,551 | 3.258 |

| Config | Input | C model state | Direct model state | M55 model state | Generic model state | RLTools model state | C ELF dec/file | Direct ELF dec/file | M55 ELF dec/file | Generic ELF dec/file | RLTools ELF dec/file |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 3 | 3,296 | 2,080 | 2,080 | 1,976 | 1,948 | 80,884/80,080 (1.000x) | 78,108/77,748 (0.966x) | 72,492/72,088 (0.896x) | 75,012/74,332 (0.927x) | 76,868/78,720 (0.950x) |
| 16x8 | 3 | 4,928 | 3,680 | 3,680 | 3,576 | 3,548 | 83,292/80,080 (1.000x) | 80,532/77,908 (0.967x) | 74,660/72,076 (0.896x) | 76,148/73,228 (0.914x) | 79,924/78,984 (0.960x) |
| 16x16 | 3 | 7,264 | 6,048 | 6,048 | 5,944 | 5,916 | 86,788/80,080 (1.000x) | 84,020/77,788 (0.968x) | 77,860/71,588 (0.897x) | 81,732/75,172 (0.942x) | 84,204/80,260 (0.970x) |
| 32x16 | 3 | 12,576 | 11,296 | 11,296 | 11,192 | 11,164 | 94,660/80,048 (1.000x) | 91,876/77,948 (0.971x) | 85,356/71,448 (0.902x) | 90,052/75,828 (0.951x) | 91,236/78,864 (0.964x) |
| 32x32 | 3 | 21,344 | 20,128 | 20,128 | 20,024 | 19,996 | 107,788/80,112 (1.000x) | 105,036/77,820 (0.974x) | 98,828/71,556 (0.917x) | 102,972/75,428 (0.955x) | 103,236/78,100 (0.958x) |
| 64x32 | 3 | 40,160 | 38,816 | 38,816 | 38,712 | 38,684 | 135,820/80,112 (1.000x) | 132,852/77,820 (0.978x) | 127,788/72,792 (0.941x) | 132,836/77,524 (0.978x) | 134,292/80,640 (0.989x) |

Raw per-variant rows, `.size.txt` paths, and ELF paths are in the CSV.
