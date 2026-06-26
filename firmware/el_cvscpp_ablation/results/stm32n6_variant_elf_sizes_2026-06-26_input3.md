# STM32N6 EL_C_vsCpp Per-Variant Sweep - 2026-06-26

Build: one firmware ELF per variant and per network size, all with `-Ofast`.
Network input features: `3`.
Runtime: Adam, 10 seeds, 2 warm-up runs per seed/variant, batch 256, 1024 rollout samples, 2 epochs, 8 optimizer steps.
Timing excludes setup, warm-up, convergence tracing, export, and numerical comparisons.
Runtime columns use per-variant logs when available, otherwise the combined `variant=all` sweep.
ELF columns report `arm-none-eabi-size dec` and on-disk ELF bytes.
Each ELF still includes the common benchmark harness and static rollout buffers.
All runtime sources completed with `status=0`: `1`.
Model-size metadata fallback: `firmware/el_cvscpp_ablation/results/stm32n6_sweep_2026-06-26_input3_10seed.csv`.

| Config | Input | C cycles | Direct C-backend cycles | Direct/C | C++ M55 cycles | M55/C | C++ Generic cycles | Generic/C | RLTools Generic cycles | RLTools/C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 3 | 5,012,812 | 4,768,041 | 0.951 | 2,288,928 | 0.457 | 2,208,650 | 0.441 | 3,114,157 | 0.621 |
| 16x8 | 3 | 6,467,890 | 6,310,011 | 0.976 | 4,083,832 | 0.631 | 4,176,785 | 0.646 | 5,550,388 | 0.858 |
| 16x16 | 3 | 8,233,737 | 8,161,413 | 0.991 | 5,804,873 | 0.705 | 6,768,004 | 0.822 | 9,836,038 | 1.195 |
| 32x16 | 3 | 11,810,771 | 11,869,999 | 1.005 | 10,367,357 | 0.878 | 12,166,980 | 1.030 | 25,064,187 | 2.122 |
| 32x32 | 3 | 17,274,911 | 21,288,642 | 1.232 | 16,849,953 | 0.975 | 21,699,226 | 1.256 | 51,012,819 | 2.953 |
| 64x32 | 3 | 31,451,158 | 37,372,086 | 1.188 | 27,561,777 | 0.876 | 41,223,215 | 1.311 | 109,287,868 | 3.475 |

| Config | Input | C model state | Direct model state | M55 model state | Generic model state | RLTools model state | C ELF dec/file | Direct ELF dec/file | M55 ELF dec/file | Generic ELF dec/file | RLTools ELF dec/file |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 3 | 3,296 | 2,080 | 2,080 | 1,976 | 1,948 | 80,884/80,080 (1.000x) | 78,140/77,780 (0.966x) | 72,492/72,088 (0.896x) | 74,996/74,332 (0.927x) | 76,868/78,720 (0.950x) |
| 16x8 | 3 | 4,928 | 3,712 | 3,712 | 3,608 | 3,548 | 83,292/80,080 (1.000x) | 80,452/77,812 (0.966x) | 74,692/72,076 (0.897x) | 76,164/73,196 (0.914x) | 79,924/78,984 (0.960x) |
| 16x16 | 3 | 7,264 | 6,048 | 6,048 | 5,944 | 5,916 | 86,788/80,080 (1.000x) | 84,060/77,820 (0.969x) | 77,860/71,588 (0.897x) | 81,756/75,204 (0.942x) | 84,204/80,260 (0.970x) |
| 32x16 | 3 | 12,576 | 11,360 | 11,360 | 11,256 | 11,164 | 94,660/80,048 (1.000x) | 91,844/77,852 (0.970x) | 85,420/71,448 (0.902x) | 90,164/75,892 (0.953x) | 91,236/78,864 (0.964x) |
| 32x32 | 3 | 21,344 | 20,128 | 20,128 | 20,024 | 19,996 | 107,788/80,112 (1.000x) | 105,084/77,852 (0.975x) | 98,828/71,556 (0.917x) | 102,916/75,364 (0.955x) | 103,236/78,100 (0.958x) |
| 64x32 | 3 | 40,160 | 38,944 | 38,944 | 38,840 | 38,684 | 135,820/80,112 (1.000x) | 132,988/77,820 (0.979x) | 127,916/72,792 (0.942x) | 132,964/77,524 (0.979x) | 134,292/80,640 (0.989x) |

Raw per-variant rows, `.size.txt` paths, and ELF paths are in the CSV.
