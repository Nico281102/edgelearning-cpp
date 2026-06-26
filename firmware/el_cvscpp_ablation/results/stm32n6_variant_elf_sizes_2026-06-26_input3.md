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
| 8x8 | 3 | 5,272,103 | 4,769,406 | 0.905 | 2,283,172 | 0.433 | 2,183,096 | 0.414 | 2,916,690 | 0.553 |
| 16x8 | 3 | 6,453,950 | 6,304,666 | 0.977 | 4,071,667 | 0.631 | 4,150,459 | 0.643 | 4,417,836 | 0.685 |
| 16x16 | 3 | 8,225,296 | 8,143,924 | 0.990 | 5,798,718 | 0.705 | 6,722,129 | 0.817 | 7,877,178 | 0.958 |
| 32x16 | 3 | 11,807,619 | 11,852,252 | 1.004 | 10,366,833 | 0.878 | 12,143,439 | 1.028 | 19,483,718 | 1.650 |
| 32x32 | 3 | 17,254,492 | 17,611,204 | 1.021 | 16,851,642 | 0.977 | 21,649,349 | 1.255 | 35,461,526 | 2.055 |
| 64x32 | 3 | 31,926,414 | 34,943,433 | 1.094 | 27,541,811 | 0.863 | 38,407,929 | 1.203 | 107,076,773 | 3.354 |

| Config | Input | C model state | Direct model state | M55 model state | Generic model state | RLTools model state | C ELF dec/file | Direct ELF dec/file | M55 ELF dec/file | Generic ELF dec/file | RLTools ELF dec/file |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 3 | 3,296 | 2,080 | 2,080 | 1,976 | 1,948 | 79,636/75,332 (1.000x) | 75,500/75,676 (0.948x) | 70,596/70,720 (0.886x) | 71,140/70,916 (0.893x) | 77,564/78,304 (0.974x) |
| 16x8 | 3 | 4,928 | 3,712 | 3,712 | 3,608 | 3,548 | 82,052/79,428 (1.000x) | 77,892/75,772 (0.949x) | 72,796/70,708 (0.887x) | 72,660/70,196 (0.886x) | 78,532/76,460 (0.957x) |
| 16x16 | 3 | 7,264 | 6,048 | 6,048 | 5,944 | 5,916 | 85,540/79,428 (1.000x) | 81,412/75,716 (0.952x) | 75,964/70,220 (0.888x) | 77,540/71,436 (0.906x) | 84,380/75,828 (0.986x) |
| 32x16 | 3 | 12,576 | 11,360 | 11,360 | 11,256 | 11,164 | 93,436/79,460 (1.000x) | 89,284/75,812 (0.956x) | 83,540/70,080 (0.894x) | 85,692/71,868 (0.917x) | 90,348/77,460 (0.967x) |
| 32x32 | 3 | 21,344 | 20,128 | 20,128 | 20,024 | 19,996 | 106,540/79,460 (1.000x) | 102,436/75,748 (0.961x) | 96,956/70,220 (0.910x) | 98,652/71,564 (0.926x) | 101,716/76,036 (0.955x) |
| 64x32 | 3 | 40,160 | 38,944 | 38,944 | 38,840 | 38,684 | 134,572/79,460 (1.000x) | 130,428/75,780 (0.969x) | 126,060/71,456 (0.937x) | 128,004/73,020 (0.951x) | 132,644/78,332 (0.986x) |

Raw per-variant rows, `.size.txt` paths, and ELF paths are in the CSV.
