# EL_C_vsCpp STM32N6 Ablation

This directory contains the firmware-side launcher for the C versus C++ M55
ablation. It intentionally does not vendor the legacy C EdgeLearning source:
point `EL_CVSCPP_EDGE_C_ROOT` at a private checkout in the local `.env`.
RLTools is also referenced only by local path through `EL_CVSCPP_RLTOOLS_ROOT`;
no RLTools sources are copied into this directory.

## Assumptions

- Board: STM32N6570-DK or Nucleo-style STM32N6 target booting through FSBL.
- CubeIDE project: a local copy named `EL_C_vsCpp`, usually cloned from
  `N6_RL`, with `STM32CubeIDE/Appli` and `STM32CubeIDE/FSBL` below the project
  root.
- The application `main.c` calls `el_cvscpp_ablation_run()` when
  `APP_EL_C_VSCPP_ABLATION` is defined.
- The FSBL has already been built as
  `STM32CubeIDE/FSBL/Debug/EL_C_vsCpp_FSBL.elf`.
- STM32CubeIDE provides `arm-none-eabi-gdb`, `ST-LINK_gdbserver`,
  CubeProgrammer, and the external loader
  `MX25UM51245G_STM32N6570-NUCLEO.stldr`.
- Python used for capture has `pyserial` installed.

## Local Configuration

```sh
cp firmware/el_cvscpp_ablation/.env.example firmware/el_cvscpp_ablation/.env
```

Edit `firmware/el_cvscpp_ablation/.env` with the local paths. The file is
ignored by git. Use `ls /dev/tty.usbmodem* /dev/cu.usbmodem*` to identify the
ST-LINK UART if the default serial port is different.

## Run One Case

```sh
sh firmware/el_cvscpp_ablation/flash_and_run_n6.sh --config 8x8
```

By default this builds the combined `all` firmware, which runs the C baseline
and all three C++ variants in one image so numerical comparisons are emitted in
the same UART log. To build and run one isolated variant instead:

```sh
sh firmware/el_cvscpp_ablation/flash_and_run_n6.sh --config 8x8 --variant cpp_m55
```

Valid variants are `all`, `legacy_c`, `cpp_direct_c_backend`, `cpp_m55`,
`cpp_generic`, and `rltools_generic`. The script builds one static application
ELF, flashes it through the external loader, launches the FSBL, and captures UART until the firmware
prints `DONE test=EL_C_vsCpp`.

Output files are written under:

```text
$EL_CVSCPP_PROJECT_ROOT/STM32CubeIDE/Appli/$EL_CVSCPP_BUILD_CONFIG/<INPUT>_<H1>x<H2>_1/
```

Single-variant ELF artifacts are written one level deeper:

```text
$EL_CVSCPP_PROJECT_ROOT/STM32CubeIDE/Appli/$EL_CVSCPP_BUILD_CONFIG/<INPUT>_<H1>x<H2>_1/<variant>/
```

The important artifacts are `serial.log`, the `.elf`, the `.map`, and
`*.size.txt`.

For a build-only check without touching the board:

```sh
sh firmware/el_cvscpp_ablation/flash_and_run_n6.sh --config 8x8 --skip-flash --skip-fsbl --skip-capture
```

To generate every per-variant ELF for the footprint ablation without touching
the board:

```sh
sh firmware/el_cvscpp_ablation/run_variant_size_sweep_n6.sh
```

## Full Sweep

```sh
sh firmware/el_cvscpp_ablation/run_sweep_n6.sh
```

Each firmware run measures:

- legacy C model using the legacy C M55 backend;
- C++ static model calling the same legacy C backend kernels directly;
- C++ static model using `edge::Backend::M55`;
- C++ static model using `edge::Backend::Generic`.
- RLTools C++ generic static model using only RLTools neural-network APIs.

All cases use input size 3 by default, batch size 256, one linear output neuron, static
model or arena storage, 10 deterministic seeds, and deterministic synthetic
linear-regression samples. The measured training protocol uses Adam
(`lr=1e-3`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`), a 1024-sample rollout, 2
epochs, 4 minibatches per epoch, 8 Adam optimizer steps, and 2048
sample-passes. The firmware Makefile compiles both C and C++ with `-Ofast`,
`-ffunction-sections`, `-fdata-sections`, and linker `--gc-sections`.
Override the input dimension with `EL_CVSCPP_INPUT_FEATURES` or
`flash_and_run_n6.sh --input-features N` when a non-Pendulum-shaped synthetic
regression ablation is needed.

The DWT timing window is intentionally narrow for 1:1 comparability: the
firmware pre-generates the rollout, imports the same initial parameters, resets
gradients/optimizer state, and exports parameters outside the measured region.
For every variant and seed the firmware first executes 2 full warm-up training
runs with the same dataset and initial parameters, resetting model and optimizer
state before the measured run. The reported cycle counts include only the hot
training work of the measured run: minibatch gradient reset, forward/backward
sample passes, and Adam updates. Setup, warm-up, convergence traces,
import/export, serial printing, and numerical comparisons are outside DWT.
Each seed also runs a separate equivalent profiling pass with the same initial
parameters and dataset, so the primary `cycles` field used for speedup is not
polluted by internal probes. C++ and RLTools variants split `zero_grad`,
optional `input_copy`, `forward`, `loss`, `backward`, and `adam_update`. The
legacy C API exposes forward/loss/backward through one `el_network_train_step`
call, so that path is reported as `sample_train_step` plus `zero_grad` and
`adam_update`.

The firmware also emits a convergence trace for seed 0. This is an untimed
diagnostic pass that reports minibatch MSE after each Adam update and does not
contribute to the cycle statistics.

The default network-size sweep is:

```text
8x8 16x8 16x16 32x16 32x32 64x32
```

`64x32` intentionally replaces the earlier `32x64` case. Larger previous
cases such as `64x64` and `128x64` are not part of the default run.

The UART log reports ablation comparisons for every seed:

- `legacy_c_vs_cpp_direct_c_backend`
- `legacy_c_vs_cpp_m55`
- `legacy_c_vs_cpp_generic`
- `cpp_generic_vs_rltools_generic`

It also prints model-size fields:

- `legacy_c_arena` and `legacy_c_control` for the C static arena/control state;
- `cpp_*_required_memory` for each C++ compile-time arena requirement;
- `cpp_*_model_object` for each C++ static model object size.
- `rltools_static_state` and `rltools_model_object` for the RLTools static
  runtime and model object.

The linked firmware footprint is recorded by `arm-none-eabi-size` in
`*.size.txt` next to the ELF.

For separate per-variant ELF sizes, build with
`run_variant_size_sweep_n6.sh`. Those ELF sizes are firmware-artifact sizes:
they still include the common benchmark harness and static rollout buffers in
each variant image.

## Current Measurement Snapshot

The 2026-06-26 input-3 ten-seed sweep shows the expected small-network RLTools
profile on the firmware path: RLTools generic is faster than legacy C on `8x8`
and `16x8`, then slower from `16x16` upward. The C++ direct legacy-C backend
remains close to the legacy C baseline on smaller and mid-size cases, while the
native C++ M55 backend is faster than legacy C across the measured sweep.

Runtime ratios below are variant cycles divided by legacy C cycles:

```text
hidden   direct-C-backend   cpp-m55   cpp-generic   rltools-generic
8x8              0.950       0.457        0.440             0.621
16x8             0.976       0.631        0.645             0.855
16x16            0.991       0.705        0.823             1.194
32x16            0.984       0.859        1.008             2.077
32x32            1.230       0.975        1.256             2.953
64x32            1.183       0.877        1.328             3.474
```

Model footprint is reported in the generated tables as C arena/control bytes,
C++ compile-time required-memory/object bytes, and RLTools static
runtime/model-object bytes. Per-variant ELF size is measured from separate
firmware images with `arm-none-eabi-size`.

## Report

After a sweep, generate CSV and Markdown tables with cycle averages, min/max,
model sizes, and ELF sizes:

```sh
python3 firmware/el_cvscpp_ablation/report_sweep_n6.py
```

The default report paths are:

```text
firmware/el_cvscpp_ablation/results/stm32n6_sweep_<date>_10seed.csv
firmware/el_cvscpp_ablation/results/stm32n6_sweep_<date>_10seed.md
```

Generate public CSV/SVG curves for speedup and convergence with:

```sh
python3 firmware/el_cvscpp_ablation/plot_sweep_n6.py
```

The default plot paths are:

```text
firmware/el_cvscpp_ablation/results/stm32n6_speedup_<date>.csv
firmware/el_cvscpp_ablation/results/stm32n6_speedup_<date>.svg
firmware/el_cvscpp_ablation/results/stm32n6_training_component_breakdown_<date>.csv
firmware/el_cvscpp_ablation/results/stm32n6_training_component_breakdown_<date>.svg
firmware/el_cvscpp_ablation/results/stm32n6_convergence_<date>.csv
firmware/el_cvscpp_ablation/results/stm32n6_convergence_<date>_32x32.svg
```

The plotter also adds a generated-plots section to the matching sweep Markdown
report when it exists.

Generate the ELF component-breakdown graph with:

```sh
python3 firmware/el_cvscpp_ablation/plot_elf_component_breakdown_n6.py
```

The default output paths are:

```text
firmware/el_cvscpp_ablation/results/stm32n6_elf_component_breakdown_<date>.csv
firmware/el_cvscpp_ablation/results/stm32n6_elf_component_breakdown_<date>.svg
```

This plot is based on the available per-variant ELF footprint fields:
`text`, `data`, and `bss`. It does not claim a runtime phase breakdown; that
would require additional timing probes inside the firmware loop.

After building separate variant ELF files, generate the footprint table with:

```sh
python3 firmware/el_cvscpp_ablation/report_variant_elf_sizes_n6.py
```

The default report paths are:

```text
firmware/el_cvscpp_ablation/results/stm32n6_variant_elf_sizes_<date>.csv
firmware/el_cvscpp_ablation/results/stm32n6_variant_elf_sizes_<date>.md
```
