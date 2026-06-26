# M55 Regression ELF Size

Size tool: `/Library/Developer/CommandLineTools/usr/bin/llvm-size`.

Each binary contains one static regression-sweep topology: input 32, two hidden Dense/ReLU layers, one linear output neuron, and batch size 256. Static model or arena storage is intentionally included so `.bss` reflects the deployment shape.

Legacy-C targets are split into `direct_backend` (C++ model calling the legacy C backend kernels directly) and `native_m55` (C++ model using `Backend::M55`).

| Target | Case | Text | RoData | Data | BSS | Other | Total sections | File bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test_m55_regression_16x16 | 16x16 | 10300 | 1207 | 80 | 27408 | 96 | 39091 | 38520 |
| test_m55_regression_16x8 | 16x8 | 11196 | 1206 | 80 | 22672 | 96 | 35250 | 39016 |
| test_m55_regression_32x16 | 32x16 | 10260 | 1207 | 80 | 52752 | 96 | 64395 | 38376 |
| test_m55_regression_32x32 | 32x32 | 10056 | 1207 | 80 | 70416 | 96 | 81855 | 36632 |
| test_m55_regression_64x32 | 64x32 | 10628 | 1207 | 80 | 137488 | 96 | 149499 | 36728 |
| test_m55_regression_8x8 | 8x8 | 12240 | 1205 | 80 | 12048 | 96 | 25669 | 54520 |
