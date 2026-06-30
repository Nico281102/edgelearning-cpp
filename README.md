# EdgeLearning++

EdgeLearning++ (`edgelearning-cpp`) is a C++20 training runtime for small neural
networks on constrained targets. It keeps the embedded assumptions of the
original C runtime: static memory, flat parameter buffers, sample-wise training,
and no heap allocation in the training path.

The C++ implementation uses templates and policies where they remove runtime
descriptors: model topology, tensor specs, layer sizes, precision types, backend
selection, and arena size are compile-time facts. Modern agentic AI tools make
it easier to study language features such as templates, concepts, and
`constexpr`; this project uses those features only where they give a concrete
embedded benefit. The C++ redesign rationale is summarized in
[docs/design/cpp_redesign_rationale.md](docs/design/cpp_redesign_rationale.md).

## Quick Start

```sh
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Minimal supervised training step:

```cpp
#include <array>
#include <edge/edge.hpp>

using Model = edge::Model<
    edge::InputVector<8>,
    edge::Dense<32, edge::ReLU>,
    edge::Dense<1>>;

int main() {
    edge::Trainer<Model, edge::MSE, edge::Adam> trainer(
        edge::AdamConfig{.learning_rate = 1.0e-3F});
    trainer.model().initialize(edge::InitConfig{.seed = 42U});

    std::array<float, 8> input{0.0F, 1.0F, 0.5F, -0.5F, 0.25F, 0.75F, -1.0F, 0.1F};
    std::array<float, 1> target{0.25F};
    return edge::is_ok(trainer.train_step(input, target)) ? 0 : 1;
}
```

See [examples/README.md](examples/README.md) for focused examples covering
minimal regression, static external arenas, custom layers, Conv2D + Flatten,
backend policies, and precision policies.

## Current Scope

| Component | Status | Notes |
|---|---:|---|
| Dense | Yes | Forward, backward, gradients, initialization, serialization |
| Conv2D | Yes | Direct CHW reference implementation |
| Flatten | Yes | Explicit shaped-to-vector boundary |
| Custom layer | Yes | Shape-aware sequential `Instance<InputSpec>` layers |
| Custom activation/loss/initializer | Yes | Policy-based extension points |
| Precision policy | Yes | Model-level arithmetic/storage type bundle |
| Backend policy | Yes | Generic backend plus M55 policy/fallback hooks |
| Dropout, LayerNorm, stateful recurrent layers | No | Future layer work |
| General graph model | No | Future planner work |

Layer inputs and outputs carry compile-time tensor metadata:

```cpp
using ImageModel = edge::Model<
    edge::Input<edge::CHW<1, 28, 28>>,
    edge::Conv2D<4, edge::Kernel<3>, edge::ReLU>,
    edge::Flatten,
    edge::Dense<10>>;
```

For the design details, see [docs/design/architecture.md](docs/design/architecture.md). For
current topology limits and future graph/stateful layers, see
[docs/design/limitations.md](docs/design/limitations.md).

## Static Memory

A default model owns one arena sized by `Model::required_memory`:

```cpp
static Model model;
```

For linker-controlled placement, provide the arena explicitly:

```cpp
alignas(Model::alignment)
static std::array<std::byte, Model::required_memory> arena;

static Model model{edge::external_arena(arena)};
```

The planner computes parameter, gradient, optimizer, activation, cache, and
workspace sizes from the model type. External arenas whose size is known in the
type are rejected at compile time if they are too small; alignment is checked at
construction. See [docs/design/memory_model.md](docs/design/memory_model.md).

## Policies

Backend selection is part of the model type:

```cpp
using M55Model = edge::Model<
    edge::Backend::M55,
    edge::InputVector<8>,
    edge::Dense<16, edge::ReLU>,
    edge::Dense<1>>;
```

`Backend::M55` falls back to generic code on host builds. On Cortex-M55/MVE FP32
builds, supported Dense paths can use M55 hooks. Unsupported operations compile
through the generic path unless a backend disables fallback.

Precision is also a model-level policy:

```cpp
struct DoublePrecision {
    using ParameterT = double;
    using ActivationT = double;
    using GradientT = double;
    using AccumulatorT = double;
    using OptimizerStateT = double;
    using LossT = double;
};

using DoubleModel = edge::Model<
    DoublePrecision,
    edge::InputVector<8>,
    edge::Dense<16, edge::ReLU>,
    edge::Dense<1>>;
```

For extension points, see [docs/api/custom_extensions.md](docs/api/custom_extensions.md).

## STM32N6 Preview

The checked-in STM32N6 sweep compares one static firmware ELF per topology and
variant. The ratio below is `RLTools generic cycles / EdgeLearning++ M55
cycles`; values above `1.0x` mean the M55 backend completed the same measured
training work in fewer cycles.

`EL++ M55 object` is `sizeof(Model)` for the owning static model. `RLTools
runtime state` is the static runtime bundle used by the RLTools benchmark
wrapper. The generated report also lists arena requirements, object sizes, and
ELF sections separately.

| Hidden | RLTools/M55 runtime ratio | EL++ M55 object | RLTools runtime state |
|---|---:|---:|---:|
| `8x8` | 1.36x | 2,080 B | 2,092 B |
| `16x8` | 1.33x | 3,680 B | 3,756 B |
| `16x16` | 1.69x | 6,048 B | 6,124 B |
| `32x16` | 2.42x | 11,296 B | 11,500 B |
| `32x32` | 3.03x | 20,128 B | 20,332 B |
| `64x32` | 4.08x | 38,816 B | 39,276 B |

Source report:
[benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/stm32n6_sweep_2026-06-26_input3_10seed.md](benchmarks/firmware/stm32n6/el_cvscpp_ablation/results/stm32n6_sweep_2026-06-26_input3_10seed.md).
Methodology and scripts:
[firmware/el_cvscpp_ablation/README.md](firmware/el_cvscpp_ablation/README.md).
The public firmware sweep can be reproduced without the legacy C checkout by
running the C++ and RLTools variants. Legacy-C rows in the generated report are
author measurements from a private external checkout.

## Benchmarks

Host and firmware benchmark methodology lives in
[docs/benchmarking/README.md](docs/benchmarking/README.md). The generated result files under
`benchmarks/host/` and `benchmarks/firmware/` are the authoritative source for
checked-in measurements.

Run host benchmarks with:

```sh
cmake -S . -B build -DEDGE_BUILD_BENCHMARKS=ON
cmake --build build --parallel
./build/benchmarks/benchmark_edgelearning_cpp
./build/benchmarks/benchmark_mixed_precision
./build/benchmarks/benchmark_regression_vs_c_baseline
```

## STM32 Toolchain Notes

The public core is C++20 and avoids exceptions, RTTI, heap allocation, virtual
functions, and `std::function`. For STM32CubeIDE or `arm-none-eabi-g++`, use
flags equivalent to:

```sh
arm-none-eabi-g++ -std=c++20 -ffreestanding -fno-exceptions -fno-rtti
```

Include `edgelearning-cpp/include` in the firmware project include paths.

## Provenance

The old C baseline is used only for methodology and local regression
measurement. Its source is not vendored in this repository. The independent
redesign boundary is documented in
[docs/design/independent_redesign.md](docs/design/independent_redesign.md).

## References

```bibtex
@article{Piao_2023,
   title={Enabling Large Batch Size Training for DNN Models Beyond the Memory Limit While Maintaining Performance},
   volume={11},
   ISSN={2169-3536},
   url={http://dx.doi.org/10.1109/ACCESS.2023.3312572},
   DOI={10.1109/access.2023.3312572},
   journal={IEEE Access},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Piao, Xinyu and Synn, Doangjoo and Park, Jooyoung and Kim, Jong-Kook},
   year={2023},
   pages={102981--102990}
}

@inproceedings{pispisa_reforward_2026,
  author    = {Pispisa, Gaetano and De Vita, Fabrizio and Bruneo, Dario and Giacalone, Davide and Merlino, Giovanni and Longo, Francesco},
  title     = {Re-Forward: Memory-efficient Backpropagation for Reinforcement Learning at the Edge},
  booktitle = {Workshop paper},
  year      = {2026},
  note      = {Bibliographic entry to be replaced with the official proceedings version when available}
}
```
