# STM32N6 C++/RLTools Benchmark and Private C Ablation

This directory contains the firmware-side launcher for the STM32N6 benchmark.
The public path compares EdgeLearning++ M55, EdgeLearning++ generic, and
RLTools generic with static batch-256 tensors. The legacy-C rows are a private
ablation: they are reproducible only when `EL_CVSCPP_EDGE_C_ROOT` points at an
external legacy C checkout. The legacy C source is not vendored, copied, or
published in this repository.
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
ST-LINK UART and put it in `EL_CVSCPP_SERIAL_PORT`.

`EL_CVSCPP_EDGE_C_ROOT` is optional for public runs. Set it only for `all`,
`legacy_c`, or `cpp_direct_c_backend`.

## Run One Case

```sh
sh firmware/el_cvscpp_ablation/flash_and_run_n6.sh --config 8x8
```

By default this builds the public `cpp_m55` isolated deployable variant. Choose
another public variant explicitly when needed:

```sh
sh firmware/el_cvscpp_ablation/flash_and_run_n6.sh --config 8x8 --variant rltools_generic
```

Valid variants are `all`, `legacy_c`, `cpp_direct_c_backend`, `cpp_m55`,
`cpp_generic`, and `rltools_generic`. The public variants are `cpp_m55`,
`cpp_generic`, and `rltools_generic`; `all`, `legacy_c`, and
`cpp_direct_c_backend` require `EL_CVSCPP_EDGE_C_ROOT`. The script builds one
static application ELF, flashes it through the external loader, launches the
FSBL, and captures UART until the firmware prints `DONE test=EL_C_vsCpp`.

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

## Full Sweep

```sh
sh firmware/el_cvscpp_ablation/run_sweep_n6.sh
```

The default sweep is public and builds, flashes, and measures one isolated
firmware image for each C++/RLTools variant and network size:

```text
cpp_m55 cpp_generic rltools_generic
```

Runtime cycles, component profiling, convergence traces, and ELF sizes in the
generated report therefore come from the same deployable firmware image.

To run only the combined same-firmware comparison smoke test:

```sh
EL_CVSCPP_SWEEP_VARIANTS=all sh firmware/el_cvscpp_ablation/run_sweep_n6.sh
```

To reproduce the private legacy-C ablation, set `EL_CVSCPP_EDGE_C_ROOT` in the
local `.env` and run:

```sh
EL_CVSCPP_SWEEP_VARIANTS="legacy_c cpp_direct_c_backend cpp_m55 cpp_generic rltools_generic" \
  sh firmware/el_cvscpp_ablation/run_sweep_n6.sh
```

To build every per-variant ELF for a footprint-only check without touching the
board:

```sh
sh firmware/el_cvscpp_ablation/run_sweep_n6.sh --skip-flash --skip-fsbl --skip-capture
```

The full private ablation measures:

- legacy C model using the legacy C M55 backend;
- C++ static model calling the same legacy C backend kernels directly;
- C++ static model using `edge::Backend::M55`;
- C++ static model using `edge::Backend::Generic`.
- RLTools C++ generic static model using only RLTools neural-network APIs and
  static batch-256 tensors.

All cases use input size 3 by default, batch size 256, one linear output
neuron, static model or arena storage, 10 deterministic seeds, and
deterministic synthetic linear-regression samples. EdgeLearning++ implements
batch 256 by accumulating 256 per-sample gradients before one Adam update;
RLTools uses a static `[256, input_features]` input tensor and one
forward/loss/backward/update per minibatch. The measured training protocol uses
Adam (`lr=1e-3`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`), a 1024-sample
rollout, 2 epochs, 4 minibatches per epoch, 8 Adam optimizer steps, and 2048
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

The combined `variant=all` smoke log reports ablation comparisons for every
seed:

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
`*.size.txt` next to each per-variant ELF. Those ELF sizes are
firmware-artifact sizes: they still include the common benchmark harness and
static rollout buffers in each variant image.

## Current Measurement Snapshot

The checked-in 2026-06-30 input-3 ten-seed run is recorded in:

```text
benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/stm32n6_sweep_2026-06-30_input3_10seed.md
```

That generated report is the source of record for runtime ratios, model-state
bytes, per-variant ELF sizes, status columns, artifact paths, and generated
plots. Keep summary text in this README method-focused; do not duplicate full
result tables here.

## Report

After a sweep, generate the single CSV and Markdown report with cycle averages,
min/max, model sizes, per-variant ELF sizes, and artifact paths:

```sh
python3 firmware/el_cvscpp_ablation/report_sweep_n6.py
```

By default the report includes the public C++/RLTools variants. For the private
legacy-C ablation, pass the same variant set used for the sweep:

```sh
python3 firmware/el_cvscpp_ablation/report_sweep_n6.py \
  --variants legacy_c cpp_direct_c_backend cpp_m55 cpp_generic rltools_generic
```

The default report paths are:

```text
benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/stm32n6_sweep_<date>_10seed.csv
benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/stm32n6_sweep_<date>_10seed.md
```

With the default input size this resolves to
`stm32n6_sweep_<date>_input3_10seed.{csv,md}`.

Generate public CSV/SVG curves for speedup and convergence with:

```sh
python3 firmware/el_cvscpp_ablation/plot_sweep_n6.py
```

The default plot paths are:

```text
benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/stm32n6_speedup_<date>.csv
benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/stm32n6_speedup_<date>.svg
benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/stm32n6_training_component_breakdown_<date>.csv
benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/stm32n6_training_component_breakdown_<date>.svg
benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/stm32n6_convergence_<date>.csv
benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/stm32n6_convergence_<date>_32x32.svg
```

When generated from an input-tagged sweep report, the plot paths also include
that input tag, for example `stm32n6_speedup_<date>_input3.svg`.

The plotter also adds a generated-plots section to the matching sweep Markdown
report when it exists.

Generate the ELF component-breakdown graph from the same sweep CSV with:

```sh
python3 firmware/el_cvscpp_ablation/plot_elf_component_breakdown_n6.py
```

The default output paths are:

```text
benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/stm32n6_elf_component_breakdown_<date>.csv
benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/stm32n6_elf_component_breakdown_<date>.svg
```

When generated from an input-tagged sweep report, these paths also include the
input tag.

This plot is based on the available per-variant ELF footprint fields:
`text`, `data`, and `bss`. It does not claim a runtime phase breakdown; that
would require additional timing probes inside the firmware loop.
