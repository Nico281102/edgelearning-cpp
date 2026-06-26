# STM32N6 EL_C_vsCpp Sweep - 2026-06-26 - 10 seeds

Board target: STM32N6 Cortex-M55 with MVE.
Task: deterministic linear regression, input 3, output 1, batch 256.
Protocol: Adam, rollout 1024, 2 epochs, 8 optimizer steps, 2048 sample-passes per measured run.
Warm-up: 2 full training runs per variant/seed, with model and optimizer reset before the measured run.
Timing: pre-generated rollout hot path only; setup, import/export, reset, sample generation, warm-up, traces, and comparisons are outside DWT.
Profiling: training-loop component counters are collected in a separate equivalent pass with the same initial parameters and dataset, then averaged over seeds.
Legacy C exposes `sample_train_step` as one combined forward/loss/backward component because those operations are encapsulated by the C API.
Convergence trace: seed 0, minibatch MSE after each Adam update, emitted by an untimed diagnostic pass.
Build: static C arena and static C++ model, all firmware objects compiled with `-Ofast`.

All runs completed with `DONE status=0`: `1`.
All numerical comparisons passed for every seed: `1`.

| Config | Input | Seeds | Warm-ups | Params | C M55 avg | Direct C-backend avg | Direct/C | C++ M55 avg | M55/C | C++ Generic avg | Generic/C | RLTools Generic avg | RLTools/C | C arena+ctrl | Direct req/obj | M55 req/obj | Generic req/obj | RLTools state/obj | ELF text | ELF data | ELF bss | ELF dec | ELF file |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8x8 | 3 | 10 | 2 | 113 | 5016850 | 4750684 | 0.947 | 2287993 | 0.456 | 2213466 | 0.441 | 3093011 | 0.617 | 3296 | 2048/2080 | 2048/2080 | 1952/1976 | 2092/1948 | 83364 | 132 | 41532 | 125028 | 129540 |
| 16x8 | 3 | 10 | 2 | 209 | 6471087 | 6312141 | 0.975 | 4082979 | 0.631 | 4169624 | 0.644 | 5530348 | 0.855 | 4928 | 3648/3680 | 3648/3680 | 3552/3576 | 3756/3548 | 82588 | 132 | 51932 | 134652 | 128856 |
| 16x16 | 3 | 10 | 2 | 353 | 8240886 | 8168195 | 0.991 | 5806241 | 0.705 | 6779423 | 0.823 | 9844205 | 1.195 | 7264 | 6016/6048 | 6016/6048 | 5920/5944 | 6124/5916 | 85076 | 132 | 67196 | 152404 | 131396 |
| 32x16 | 3 | 10 | 2 | 673 | 11803468 | 11859089 | 1.005 | 10366813 | 0.878 | 12175991 | 1.032 | 25041861 | 2.122 | 12576 | 11264/11296 | 11264/11296 | 11168/11192 | 11500/11164 | 84420 | 132 | 101308 | 185860 | 130784 |
| 32x32 | 3 | 10 | 2 | 1217 | 17280759 | 21242599 | 1.229 | 16850997 | 0.975 | 21755360 | 1.259 | 51007076 | 2.952 | 21344 | 20096/20128 | 20096/20128 | 20000/20024 | 20332/19996 | 83228 | 132 | 158460 | 241820 | 129524 |
| 64x32 | 3 | 10 | 2 | 2369 | 33037271 | 34738886 | 1.052 | 27555437 | 0.834 | 41032636 | 1.242 | 107631551 | 3.258 | 40160 | 38784/38816 | 38784/38816 | 38688/38712 | 39276/38684 | 89292 | 132 | 279932 | 369356 | 135664 |

Raw UART logs and `.size.txt` files are referenced in the CSV.

<!-- plots:start -->
## Generated plots

![Speedup over legacy C](stm32n6_speedup_2026-06-26_input3.svg)

![Training-loop component breakdown](stm32n6_training_component_breakdown_2026-06-26_input3.svg)

![Convergence trace](stm32n6_convergence_2026-06-26_input3_32x32.svg)
<!-- plots:end -->
