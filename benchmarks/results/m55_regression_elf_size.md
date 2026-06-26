# M55 Regression ELF Size

Size tool: `/Library/Developer/CommandLineTools/usr/bin/llvm-size`.

Each binary contains one static regression-sweep topology: input 32, two hidden Dense/ReLU layers, one linear output neuron, and batch size 256. Static model or arena storage is intentionally included so `.bss` reflects the deployment shape.

Legacy-C targets are split into `direct_backend` (C++ model calling the legacy C backend kernels directly) and `native_m55` (C++ model using `Backend::M55`).

| Target | Case | Text | RoData | Data | BSS | Other | Total sections | File bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test_legacy_c_generic_direct_backend_regression_128x64 | 128x64 | 29832 | 1588 | 296 | 306208 | 652 | 338576 | 89784 |
| test_legacy_c_generic_native_m55_regression_128x64 | 128x64 | 31028 | 1621 | 624 | 306220 | 1144 | 340637 | 108032 |
| test_m55_regression_128x64 | 128x64 | 20932 | 687 | 824 | 405664 | 1212 | 429319 | 112072 |
| test_legacy_c_generic_direct_backend_regression_16x16 | 16x16 | 29012 | 1587 | 280 | 21920 | 628 | 53427 | 88680 |
| test_legacy_c_generic_native_m55_regression_16x16 | 16x16 | 30200 | 1620 | 608 | 21932 | 1120 | 55480 | 106032 |
| test_m55_regression_16x16 | 16x16 | 19548 | 686 | 808 | 27552 | 1188 | 49782 | 108664 |
| test_legacy_c_generic_direct_backend_regression_16x8 | 16x8 | 29240 | 1586 | 296 | 18400 | 652 | 50174 | 89544 |
| test_legacy_c_generic_native_m55_regression_16x8 | 16x8 | 30432 | 1619 | 624 | 18412 | 1144 | 52231 | 107616 |
| test_m55_regression_16x8 | 16x8 | 19780 | 685 | 824 | 22880 | 1212 | 45381 | 111304 |
| test_legacy_c_generic_direct_backend_regression_32x16 | 32x16 | 29096 | 1587 | 280 | 40864 | 628 | 72455 | 89192 |
| test_legacy_c_generic_native_m55_regression_32x16 | 32x16 | 30284 | 1620 | 608 | 40876 | 1120 | 74508 | 107296 |
| test_m55_regression_32x16 | 32x16 | 19712 | 686 | 808 | 52768 | 1188 | 75162 | 111160 |
| test_legacy_c_generic_direct_backend_regression_32x32 | 32x32 | 28024 | 1587 | 248 | 54048 | 580 | 84487 | 87080 |
| test_legacy_c_generic_native_m55_regression_32x32 | 32x32 | 28988 | 1620 | 544 | 54060 | 1024 | 86236 | 102064 |
| test_m55_regression_32x32 | 32x32 | 17816 | 686 | 728 | 70304 | 1068 | 90602 | 102728 |
| test_legacy_c_generic_direct_backend_regression_32x64 | 32x64 | 29308 | 1587 | 280 | 80928 | 628 | 112731 | 89224 |
| test_legacy_c_generic_native_m55_regression_32x64 | 32x64 | 30496 | 1620 | 608 | 80940 | 1120 | 114784 | 107344 |
| test_m55_regression_32x64 | 32x64 | 20136 | 686 | 808 | 105888 | 1188 | 128706 | 111208 |
| test_legacy_c_generic_direct_backend_regression_64x64 | 64x64 | 29588 | 1587 | 280 | 155680 | 628 | 187763 | 88744 |
| test_legacy_c_generic_native_m55_regression_64x64 | 64x64 | 30780 | 1620 | 608 | 155692 | 1120 | 189820 | 106128 |
| test_m55_regression_64x64 | 64x64 | 20684 | 686 | 808 | 205472 | 1188 | 228838 | 108808 |
| test_legacy_c_generic_direct_backend_regression_8x8 | 8x8 | 29004 | 1585 | 280 | 10464 | 628 | 41961 | 88544 |
| test_legacy_c_generic_native_m55_regression_8x8 | 8x8 | 30192 | 1618 | 608 | 10476 | 1120 | 44014 | 105792 |
| test_m55_regression_8x8 | 8x8 | 19536 | 684 | 808 | 12320 | 1188 | 34536 | 108264 |
