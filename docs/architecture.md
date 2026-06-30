# Architecture

EdgeLearning++ is a C++20 training runtime for small neural networks on
embedded targets. A model is described as a C++ type, and that type carries the
information needed to derive shapes, parameter counts, activation storage,
temporary workspace, and backend dispatch.

```cpp
using Model = edge::Model<
    edge::InputVector<8>,
    edge::Dense<32, edge::ReLU>,
    edge::Dense<1>>;
```

This alias does not allocate memory. It names one concrete model type. When a
`Model` object is created, the type already knows its input size, output size,
layer count, memory requirement, alignment requirement, precision types, and
backend policy.

## Design Principles

The C++ architecture keeps the embedded constraints of the original C runtime:
deterministic memory ownership, flat buffers, no heap allocation in the training
path, explicit status reporting, and small kernels that can be inspected in
generated code.

The C++ layer adds static structure on top of those constraints:

- model topology is a type, not a runtime descriptor
- layers are compile-time policies, not virtual objects
- tensor shape and layout are type-level metadata
- memory sizes and section offsets are `static constexpr` constants
- backend selection is a compile-time policy
- unsupported strict backend paths fail at compile time
- runtime buffers are passed through typed views with fixed extents

C++ is used here to expose more constants to the compiler while keeping the
runtime representation close to the embedded C baseline: contiguous storage,
direct pointer arithmetic, static dispatch, and explicit arenas.

## Model Type

`edge::Model` accepts optional policies followed by an input spec and a sequence
of layer specs:

```cpp
using DefaultModel = edge::Model<
    edge::InputVector<8>,
    edge::Dense<16, edge::ReLU>,
    edge::Dense<1>>;

using M55Model = edge::Model<
    edge::Backend::M55,
    edge::InputVector<8>,
    edge::Dense<16, edge::ReLU>,
    edge::Dense<1>>;

using MixedPrecisionModel = edge::Model<
    edge::precision::MixedFP16,
    edge::InputVector<8>,
    edge::Dense<16, edge::ReLU>,
    edge::Dense<1>>;
```

The accepted forms are:

- `Model<Input, Layers...>`
- `Model<Precision, Input, Layers...>`
- `Model<Backend, Input, Layers...>`
- `Model<Backend, Precision, Input, Layers...>`

Internally, `Model` normalizes those arguments into one `ModelImpl` with four
compile-time inputs: backend policy, precision policy, input layer, and layer
type list.

The public model type exposes the main facts derived from that list:

- `input_spec` and `output_spec`
- `input_size` and `output_size`
- `parameter_count`, `gradient_count`, `optimizer_state_count`
- `activation_count`, `cache_count`, `workspace_count`
- `parameter_bytes`, `gradient_bytes`, `optimizer_bytes`
- `activation_bytes`, `workspace_bytes`, `required_memory`
- `alignment`
- typed aliases such as `parameter_type`, `activation_type`, `gradient_type`

Those values are available at compile time and can be used in `static_assert`,
array extents, linker-placed buffers, and firmware build checks.

## Tensor Vocabulary

The tensor vocabulary separates compile-time semantic metadata from runtime
storage.

### Shape

`edge::Shape<Dims...>` is a compile-time list of positive dimensions:

```cpp
using ImageShape = edge::Shape<1, 28, 28>;
static_assert(ImageShape::rank == 3);
static_assert(ImageShape::elements == 784);
```

`Shape` stores no runtime data. It gives layers a way to reason about rank and
extent before code is generated.

### Layout

`edge::Layout` describes what the dimensions mean:

```cpp
enum class Layout {
    Flat,
    CHW,
    HWC,
    TimeFeature,
    TokenFeature,
    HeadTokenFeature
};
```

`Shape<32, 64>` alone is ambiguous. It could mean time-feature, token-feature,
or a simple matrix-like internal representation. `Layout` gives custom layers a
shared vocabulary for shape-aware validation.

The current runtime implements sequential single-input/single-output models.
The extra layout names are metadata only. They let future or custom layers reject
incompatible tensors at compile time without adding a dynamic tensor descriptor.

### TensorSpec

`edge::TensorSpec<ShapeT, LayoutV>` binds shape and layout:

```cpp
using Image = edge::TensorSpec<edge::Shape<1, 28, 28>, edge::Layout::CHW>;
using Tokens = edge::TensorSpec<edge::Shape<16, 128>, edge::Layout::TokenFeature>;
```

The library provides aliases for common cases:

```cpp
using Vec = edge::Vector<8>;        // TensorSpec<Shape<8>, Flat>
using Img = edge::CHW<1, 28, 28>;   // TensorSpec<Shape<1, 28, 28>, CHW>
using Img2 = edge::HWC<28, 28, 1>;  // TensorSpec<Shape<28, 28, 1>, HWC>
```

Every layer instance exposes:

```cpp
using input_spec = ...;
using output_spec = ...;
```

`Model` chains those specs. The output spec of one layer becomes the input spec
of the next layer.

### Input and InputVector

`edge::Input<TensorSpecT>` marks the external model input:

```cpp
using Model = edge::Model<
    edge::Input<edge::CHW<1, 28, 28>>,
    edge::Conv2D<4, edge::Kernel<3>, edge::ReLU>,
    edge::Flatten,
    edge::Dense<10>>;
```

For vector models, `InputVector<N>` is the short form:

```cpp
using Model = edge::Model<
    edge::InputVector<3>,
    edge::Dense<8, edge::Tanh>,
    edge::Dense<1>>;
```

`Input` is not a layer with runtime behavior. It is the starting tensor spec for
compile-time shape propagation.

## TensorView

`edge::TensorView<T, N>` is the runtime buffer view used by kernels:

```cpp
edge::TensorView<const float, 8> input_view(input_array);
edge::TensorView<float, 1> output_view(output_array);
```

It contains only a pointer. The extent `N` is part of the type, so a layer
receives fixed-size views such as `TensorView<const ActivationT, in_features>`
and `TensorView<GradientT, parameter_count>`.

`TensorView` does not carry `Shape` or `Layout`. Shape and layout are
compile-time model metadata; kernels operate on contiguous flattened storage.
Layer definitions still validate semantic shape at compile time.

Constness is also part of the view type:

- `TensorView<const T, N>` means read-only
- `TensorView<T, N>` means writable

The view can be constructed from fixed-size arrays and `std::array` objects.
The model checks null views at runtime and returns `Status::NullPointer`.

`edge::StaticTensor<T, N>` is a small owning helper around
`std::array<T, N>`. It is useful for examples and tests, but the model itself
uses arena-backed storage.

## Layer Specs and Layer Instances

A public layer type is a layer specification. It describes user intent without
knowing the previous layer shape yet:

```cpp
edge::Dense<32, edge::ReLU>
edge::Conv2D<4, edge::Kernel<3>, edge::ReLU>
edge::Flatten
```

During model instantiation, each layer spec becomes an input-specialized layer
instance. Conceptually:

```cpp
using instance = typename MakeLayerInstance<CurrentSpec, LayerSpec>::type;
using next_spec = typename instance::output_spec;
```

The instance exposes a uniform contract:

```cpp
using input_spec = ...;
using output_spec = ...;

static constexpr std::size_t in_features = input_spec::elements;
static constexpr std::size_t out_features = output_spec::elements;
static constexpr std::size_t parameter_count = ...;
static constexpr std::size_t cache_count = ...;
static constexpr std::size_t workspace_count = ...;
```

It also provides three static functions:

```cpp
initialize(params, rng, config);
forward(input, output, params, cache, workspace);
backward<PropagateInputGradient>(
    input, output, upstream, downstream, params, gradients, cache, workspace);
```

There is no mandatory separate `backward_input_gradient(...)` function. Input
gradient propagation is part of `backward` and is controlled by the
`PropagateInputGradient` template boolean. This keeps the layer contract smaller
and lets the compiler remove downstream-gradient code when it is not needed.

## Built-In Layers

### Dense

`Dense<OutFeatures, Activation>` requires a flat input spec:

```cpp
using Model = edge::Model<
    edge::InputVector<8>,
    edge::Dense<32, edge::ReLU>,
    edge::Dense<1>>;
```

If the previous layer is shaped, insert `Flatten` before `Dense`:

```cpp
using Model = edge::Model<
    edge::Input<edge::CHW<1, 28, 28>>,
    edge::Conv2D<4, edge::Kernel<3>, edge::ReLU>,
    edge::Flatten,
    edge::Dense<10>>;
```

The compile-time assertion avoids implicit layout loss. A shaped tensor may have
the same number of elements as a vector but a different meaning, so `Flatten`
marks the semantic boundary explicitly.

### Conv2D

`Conv2D` currently accepts CHW input:

```cpp
edge::Conv2D<
    4,                 // output channels
    edge::Kernel<3>,   // kernel height and width
    edge::ReLU,
    edge::DefaultInitializer,
    edge::Stride<1>,
    edge::Padding<0>>
```

The layer derives input channels, height, and width from the incoming
`TensorSpec`. Output height and width are compile-time constants derived from
kernel, stride, and padding.

### Flatten

`Flatten` maps any input spec to `Vector<InputSpec::elements>`.

It has no parameters, no cache, and no layer-local workspace. Forward copies the
flattened input to the flattened output. Backward copies the upstream gradient
only when `PropagateInputGradient` is `true`.

Flatten is a layer rather than an implicit conversion because downstream layers
no longer see spatial, temporal, or token structure after that point.

## Forward Pass

The model stores the external input and every layer output in one activation
section. Forward execution is a compile-time recursion over the layer instance
list.

For each layer, the recursion computes these offsets as template constants:

- `ParamOffset`: first parameter element for this layer
- `ActOutOffset`: first activation element where this layer writes its output
- `CacheOffset`: first cache element for this layer
- `WorkspaceOffset`: first layer-local temporary workspace element

The generated call is equivalent to passing slices of the flat arena:

```cpp
Instance::forward(
    TensorView<const ActivationT, Instance::in_features>(previous),
    TensorView<ActivationT, Instance::out_features>(activations + ActOutOffset),
    TensorView<const ParameterT, Instance::parameter_count>(parameters + ParamOffset),
    TensorView<ActivationT, Instance::cache_count>(cache_base + CacheOffset),
    TensorView<AccumulatorT, Instance::workspace_count>(workspace + WorkspaceOffset));
```

Because the offsets are template constants, the compiler can fold most address
arithmetic and inline the layer path.

## Backward Pass

Backward execution first recurses to the last layer, then unwinds in reverse.
Each layer receives:

- the previous activation
- this layer output
- upstream gradient, `dLoss/dOutput`
- optional downstream gradient, `dLoss/dInput`
- this layer parameters
- this layer gradient slice
- this layer cache
- this layer-local workspace

The important backward offsets are:

- `ParamOffset`: first parameter element for this layer
- `GradOffset`: first gradient element for this layer
- `ActOutOffset`: first activation element for this layer output
- `PrevActOffset`: first activation element for this layer input
- `CacheOffset`: first cache element for this layer
- `WorkspaceOffset`: first layer-local workspace element

`PrevActOffset` is not another allocation. It points into the same activation
section. For the first real layer it points to the stored external input. For
later layers it points to the previous layer output.

The model uses two accumulator slots, often described as slot 0 and slot 1. One
slot holds the current upstream gradient. The other slot is available for the
downstream gradient that must feed the previous layer. After each layer, the two
pointers are swapped.

The first real layer uses:

```cpp
static constexpr bool propagate_input_gradient = LayerIndex != 0;
```

So ordinary training does not allocate or compute `dLoss/dExternalInput`.
Internal layers receive `PropagateInputGradient=true`, because their downstream
gradient is needed by the layer before them.

The current design therefore does not need a mandatory
`backward_input_gradient` hook. If a future API needs gradients with respect to
external inputs for saliency, adversarial methods, or differentiable
environments, the model-level API can expose that as an explicit mode without
forcing every layer to implement a second backward function.

## Static Memory Planning

The model owns one contiguous arena by default:

```cpp
Model model;
```

The internal arena is a `std::array<std::byte, Model::required_memory>` member
of the model object. On firmware, large models should normally be static:

```cpp
static Model model;
```

When memory placement matters, use an external arena:

```cpp
alignas(Model::alignment)
static std::array<std::byte, Model::required_memory> arena;

static Model model{edge::external_arena(arena)};
```

The arena is split into sections:

- parameters
- gradients
- optimizer state
- activations and activation caches
- backward slots and layer-local workspace

Section sizes and offsets are computed from layer traits and precision policy
types. Metadata such as shape, layout, backend policy, layer count, and offsets
is not stored in the arena; it exists in types and constants.

See `docs/memory_model.md` for the detailed formulas and external arena rules.

## Precision Policy

Precision is a model-level policy. The default is `edge::precision::FP32`:

```cpp
struct FP32 {
    using ParameterT = float;
    using ActivationT = float;
    using GradientT = float;
    using AccumulatorT = float;
    using OptimizerStateT = float;
    using LossT = float;
};
```

These aliases feed both the runtime API and the memory planner. For example,
`Model::parameter_data()` returns `ParameterT*`, `Model::forward()` accepts
`ActivationT`, and `workspace_bytes` uses `sizeof(AccumulatorT)`.

The precision policy is intentionally separate from backend policy. A backend
selects implementation paths. A precision policy selects storage and arithmetic
types.

## Backend Policy

Backend selection is a compile-time model policy:

```cpp
using Model = edge::Model<
    edge::Backend::M55,
    edge::InputVector<8>,
    edge::Dense<16, edge::ReLU>,
    edge::Dense<1>>;
```

A backend policy exposes:

```cpp
static constexpr bool is_backend_policy = true;
static constexpr const char* name = "...";
```

It may also expose:

```cpp
static constexpr bool falls_back_to_generic = true;
```

If `falls_back_to_generic` is absent, the library treats it as `true`. That is
the permissive default for user backends. If a backend sets it to `false`, then
a layer must provide a compatible backend path or fail to compile.

Fallback behavior is part of the backend policy:

- `Backend::Generic` uses generic scalar C++ paths
- `Backend::M55` may use optimized M55/MVE paths when target and precision match
- `Backend::M55` can still compile on host because it falls back to generic code
- a strict backend can disable fallback to catch missing optimized kernels

Activation policies are semantic policies, not backend policies. The user writes
`Dense<32, ReLU>`; the selected backend may specialize that operation if it has a
compatible implementation.

## Training Surface

`edge::Trainer<Model, Loss, Optimizer>` is a small convenience layer around:

1. `model.forward(input)`
2. loss evaluation and output-gradient construction
3. `model.backward(output_gradient)`
4. optimizer step after the configured batch has accumulated

The trainer does not own a dynamic graph and does not allocate per step. It owns
or references a model, stores one output-gradient buffer, and accumulates
gradients in the model gradient section until `flush()`.

The model can also be driven manually when firmware code needs tighter control
over batching, profiling, or custom loss computation.

## Extensibility

The primary extension points are:

- custom activations
- custom losses
- custom initializers
- custom precision policies
- custom backend policies
- custom sequential layers

Custom sequential layers should provide `template<typename InputSpec> struct
Instance`, expose `input_spec` and `output_spec`, and implement the standard
`initialize`, `forward`, and templated `backward` functions.

For shape-aware layers, prefer explicit compile-time validation:

```cpp
static_assert(InputSpec::layout == edge::Layout::TokenFeature);
static_assert(InputSpec::rank == 2);
```

This is the intended use of the shape/layout vocabulary. The library does not
need a dynamic tensor object to let custom layers express layout requirements.

The current topology is sequential and single-input/single-output. Structures
such as residual edges, multi-input layers, recurrent state, and attention masks
need a graph planner. `TensorSpec` is useful for that future work, but it is not
a dynamic tensor IR.

## Project Layout

The directory layout follows a C++ header-oriented library structure:

- `include/edge/`: public template API and header-only runtime core
- `include/edge/layers/`: built-in layer specs and layer concepts
- `include/edge/backends/`: target-specific backend hooks
- `include/edge/optimizers/`: optimizer policies
- `tests/`: host CTest executables
- `examples/`: small user-facing examples
- `benchmarks/`: host timing and regression methodology
- `firmware/`: embedded ablation and launch scripts
- `docs/`: architecture, memory, error, extension, and benchmarking notes

Most core code is header-only because the model topology is encoded in template
arguments. The compiler must see the layer definitions to infer shapes, inline
paths, remove unused fallback branches, and compute memory requirements.

## Layer Support

| Layer or extension | Supported in v0.1 | Notes |
|---|---:|---|
| Dense | Yes | Flat input only; insert Flatten after shaped layers |
| Conv2D | Yes | Direct CHW convolution with stride and padding |
| Flatten | Yes | Explicit shaped-to-vector boundary |
| Custom layer | Yes | Sequential, shape-aware custom layers |
| Custom activation | Yes | User policies supported |
| Custom loss | Yes | User policies supported |
| Custom precision policy | Yes | Model-level policy supported |
| Custom backend policy | Yes | Compile-time policy with optional generic fallback |
| Dropout | No | Planned extension |
| LayerNorm | No | Planned extension |
| Multi-input graph layers | No | Future planner work |
| Stateful recurrent layers | No | Future planner work |
