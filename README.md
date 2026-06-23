# EdgeLearning++

EdgeLearning++ (`edgelearning-cpp`) is a clean C++20 redesign of the pre-internship EdgeLearning C core for static-memory embedded and host targets. It uses compile-time topology, shape inference, backend tags, activation policies, memory planning, and optimizer selection so common model errors fail at compile time.

The v0.1 implementation is Dense-only and fully supports FP32 on the generic scalar backend. It does not include PPO, reinforcement learning applications, CarRacing, Pendulum, STM32N6 application code, STAI integration, host-MCU protocols, private datasets, generated models, or post-baseline optimized kernels.

## Build

```sh
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The CMake target `edge::edgelearning` is header-only. For GNU/Clang builds, CMake applies `-fno-exceptions` and `-fno-rtti`. On AppleClang installations where libc++ headers live only in the macOS SDK, CMake adds the SDK `usr/include/c++/v1` path when detected.

## Define A Model

```cpp
#include <edge/edge.hpp>

using Model = edge::Model<
    edge::Input<8>,
    edge::Dense<32, edge::ReLU>,
    edge::Dense<16, edge::ReLU>,
    edge::Dense<1>>;

static_assert(Model::parameter_bytes > 0);
static_assert(Model::required_memory > 0);
```

The input dimension of each Dense layer is inferred from the previous layer. Backend selection is model-level:

```cpp
using M = edge::Model<
    edge::Backend::Generic,
    edge::Input<8>,
    edge::Dense<32, edge::ReLU>,
    edge::Dense<1>>;
```

`Backend::M55` is a tag placeholder in v0.1 and falls back to the generic implementation. Vendor STM32 headers are not included by public generic headers.

## Memory Planning

The default model owns one static arena:

```cpp
Model model;
```

The model exposes:

```cpp
Model::parameter_bytes;
Model::gradient_bytes;
Model::optimizer_bytes;
Model::activation_bytes;
Model::workspace_bytes;
Model::total_bytes;
Model::required_memory;
Model::alignment;
```

External arenas are checked at compile time because the size is part of the type:

```cpp
alignas(Model::alignment)
static std::array<std::byte, Model::required_memory> arena;

Model model{edge::external_arena(arena)};
```

`std::span<std::byte, N>` and `std::byte (&)[N]` are also supported. A future `void* + size_t` API, if added, must return `Status` and be documented as runtime-checked only.

## Training

```cpp
edge::Trainer<Model, edge::MSE, edge::Adam> trainer(
    edge::AdamConfig{.learning_rate = 1.0e-3F});
trainer.model().initialize(edge::InitConfig{.seed = 42U});

std::array<float, 8> input{};
std::array<float, 1> target{};
edge::Status status = trainer.train_step(input, target);
```

Training is sample-wise. Gradients accumulate until `TrainerConfig::batch_size` samples are seen, then the optimizer step runs. `GradientReduction::Mean` is the default and divides once before the optimizer step. Call `flush()` to update on an incomplete final batch.

The low-level API also supports:

```cpp
model.forward(input);
model.backward(output_gradient);
```

## Custom Loss

Custom losses provide `value` and `evaluate`:

```cpp
struct MyLoss {
    template<typename Prediction, typename Target>
    static float value(const Prediction& y, const Target& t);

    template<typename Prediction, typename Target, typename Gradient>
    static float evaluate(const Prediction& y, const Target& t, Gradient& g);
};
```

## Custom Activation

```cpp
struct MyActivation {
    static constexpr edge::ActivationStorage storage =
        edge::ActivationStorage::OutputAndPreActivation;

    template<typename T>
    static T forward(T z);

    template<typename T>
    static T derivative(T z, T a);
};
```

Built-ins are `Linear`, `ReLU`, `Tanh`, and `Sigmoid`.

## Benchmarks

```sh
cmake -S . -B build -DEDGE_BUILD_BENCHMARKS=ON
cmake --build build --parallel
./build/benchmarks/benchmark_edgelearning_cpp
./build/benchmarks/benchmark_regression_vs_c_baseline
```

The regression benchmark generates `benchmarks/results/host_regression_report.md`. It records methodology for comparing against the old C baseline at commit `0085814908ca1b57ece4fe367361d084fd74aa3e` without vendoring or republishing that C source.

## STM32 Toolchain Notes

The core is C++20 and avoids exceptions, RTTI, heap allocation, virtual functions, and `std::function`. For STM32CubeIDE or `arm-none-eabi-g++`, use flags equivalent to:

```sh
arm-none-eabi-g++ -std=c++20 -ffreestanding -fno-exceptions -fno-rtti
```

Include `edgelearning-cpp/include` in the CubeIDE project include paths. Firmware execution tests are outside v0.1.

