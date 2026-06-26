# STM32N6 EL_C_vsCpp Sweep - 2026-06-26 - 10 seeds

Board target: STM32N6 Cortex-M55 with MVE.  
Task: deterministic linear regression, input 3, output 1, batch 256.  
Protocol: Adam, rollout 1024, 2 epochs, 8 optimizer steps, 2048 sample-passes per measured run.  
Warm-up: 2 full training runs per variant/seed, with model and optimizer reset before the measured run.  
Timing: pre-generated rollout hot path only; setup, import/export, reset, sample generation, warm-up, traces, and comparisons are outside DWT.  
Convergence trace: seed 0, minibatch MSE after each Adam update, emitted by an untimed diagnostic pass.  
Build: static C arena and static C++ model, all firmware objects compiled with `-Ofast`.

All runs completed with `DONE status=0`: `1`.  
All numerical comparisons passed for every seed: `1`.

| Config | Input | Seeds | Warm-ups | Params | C M55 avg | Direct C-backend avg | Direct/C | C++ M55 avg | M55/C | C++ Generic avg | Generic/C | RLTools Generic avg | RLTools/C | C arena+ctrl | Direct req/obj | M55 req/obj | Generic req/obj | RLTools state/obj | ELF text | ELF data | ELF bss | ELF dec | ELF file |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 3 | 10 | 2 | 113 | 5272103 | 4769406 | 0.905 | 2283172 | 0.433 | 2183096 | 0.414 | 2916690 | 0.553 | 3296 | 2048/2080 | 2048/2080 | 1952/1976 | 2092/1948 | 77316 | 132 | 41532 | 118980 | 119892 |
| 16x8 | 3 | 10 | 2 | 209 | 6453950 | 6304666 | 0.977 | 4071667 | 0.631 | 4150459 | 0.643 | 4417836 | 0.685 | 4928 | 3680/3712 | 3680/3712 | 3584/3608 | 3756/3548 | 74700 | 132 | 52028 | 126860 | 117300 |
| 16x16 | 3 | 10 | 2 | 353 | 8225296 | 8143924 | 0.990 | 5798718 | 0.705 | 6722129 | 0.817 | 7877178 | 0.958 | 7264 | 6016/6048 | 6016/6048 | 5920/5944 | 6124/5916 | 77780 | 132 | 67196 | 145108 | 121108 |
| 32x16 | 3 | 10 | 2 | 673 | 11807619 | 11852252 | 1.004 | 10366833 | 0.878 | 12143439 | 1.028 | 19483718 | 1.650 | 12576 | 11328/11360 | 11328/11360 | 11232/11256 | 11500/11164 | 76204 | 132 | 101500 | 177836 | 119508 |
| 32x32 | 3 | 10 | 2 | 1217 | 17254492 | 17611204 | 1.021 | 16851642 | 0.977 | 21649349 | 1.255 | 35461526 | 2.055 | 21344 | 20096/20128 | 20096/20128 | 20000/20024 | 20332/19996 | 74636 | 132 | 158460 | 233228 | 117860 |
| 64x32 | 3 | 10 | 2 | 2369 | 31926414 | 34943433 | 1.094 | 27541811 | 0.863 | 38407929 | 1.203 | 107076773 | 3.354 | 40160 | 38912/38944 | 38912/38944 | 38816/38840 | 39276/38684 | 79700 | 132 | 280316 | 360148 | 122992 |

Raw UART logs and `.size.txt` files are referenced in the CSV.
