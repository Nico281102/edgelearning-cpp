# Memory Model

EdgeLearning++ uses one contiguous arena in v0.1. A default `Model` owns an internal arena sized by `Model::required_memory`.

`Model::required_memory` is a `static constexpr std::size_t`. It is computed while the model type is instantiated, so it can be used in `static_assert`, array extents, `alignas` declarations, linker-placed buffers, and other compile-time contexts.

The internal arena is a member of the `Model` object. Its storage therefore follows the object:

```cpp
Model local_model;        // internal arena is on the current stack
static Model static_model; // internal arena is in static storage, normally .bss
```

On embedded and RTOS targets, avoid placing large models on task stacks. Prefer `static Model model;` for simple firmware, or an explicitly declared static external arena when placement matters:

```cpp
alignas(Model::alignment)
static std::array<std::byte, Model::required_memory> arena;

Model model{edge::external_arena(arena)};
```

The second form keeps the large buffer in static storage even if the `Model` handle itself is local. It also makes linker placement straightforward with a target-specific section attribute.

The compile-time memory breakdown is:

- `parameter_bytes`: weights and biases
- `gradient_bytes`: accumulated parameter gradients
- `optimizer_bytes`: reserved optimizer state, large enough for Adam moments
- `activation_bytes`: input, layer outputs, and requested pre-activations
- `workspace_bytes`: temporary backward slots and layer-local workspace
- `total_bytes`: bytes through the last section, including inter-section alignment padding
- `required_memory`: aligned arena size

Metadata such as shapes, layer count, Conv2D output geometry, offsets, and policies is represented in types and constants, not stored in the arena.

## Static Memory Planning

The technique used here is best described as static memory planning or compile-time arena sizing. It is not a C++ standard library component by itself; it is a design pattern built from C++20 language features:

- template parameters carry model structure and layer sizes in the type system
- `static constexpr` members expose counts and byte sizes as compile-time constants
- `sizeof` and `alignof` convert precision policy types into byte and alignment requirements
- `alignas` lets user storage request the alignment required by the model
- `std::array<std::byte, N>` creates fixed-size storage where `N` must be known at compile time
- `static_assert` turns invalid configurations into compile errors

The `static_assert` examples in this document are only documentation and user checks. They are not required in normal application code. The framework already uses the same compile-time constants internally, and external arenas are checked by the `Model` constructor when their size is part of the type.

The planner works because the C++ API describes the network as a type:

```cpp
using Model = edge::Model<
    edge::Input<edge::CHW<1, 28, 28>>,
    edge::Conv2D<4, edge::Kernel<3, 3>, edge::ReLU>,
    edge::Flatten,
    edge::Dense<10>>;
```

When this alias is instantiated, the compiler sees the full topology. `using Model = ...` creates a type alias, not an object. No model memory is allocated by the alias itself; it simply names a concrete model type.

Each layer specification creates a typed layer instance for the input size produced by the previous layer. That instance exposes constants such as `out_features`, `parameter_count`, `cache_count`, and `workspace_count`. The model recursively accumulates those constants across the chain.

Backward propagation uses two accumulator slots that are reused in ping-pong fashion. For each layer, one slot holds `dLoss/dLayerOutput` and the other slot holds `dLoss/dLayerInput` only when the model asks that layer to propagate an input gradient. During ordinary training the first layer receives `PropagateInputGradient=false`, so the planner does not reserve a downstream buffer for the external input sample.

The important C++ pieces are:

```cpp
template<typename InputSpec, typename DenseSpec>
struct DenseInstance {
    using input_spec = InputSpec;
    using output_spec = edge::Vector<DenseSpec::out_features>;

    static constexpr std::size_t in_features = input_spec::elements;
    static constexpr std::size_t out_features = output_spec::elements;
    static constexpr std::size_t parameter_count =
        in_features * out_features + out_features;
};
```

`template<typename InputSpec, ...>` means the layer receives shape and layout as a type, not as a runtime descriptor. `static constexpr` means the values belong to the type and can be evaluated at compile time. So `DenseInstance<Vector<8>, Dense<32>>::parameter_count` is a compile-time fact.

The model chain then folds those facts:

```cpp
using instance = typename MakeLayerInstance<CurrentSpec, Layer>::type;
using tail = LayerChain<typename instance::output_spec, Rest...>;

static constexpr std::size_t parameter_count =
    instance::parameter_count + tail::parameter_count;
```

`typename` is needed because `MakeLayerInstance<...>::type` is a nested type selected by a template. `using tail = ...` names the rest of the network. The recursive definition means the compiler walks the model at compile time: current layer first, then the remaining layers.

Precision is part of the same calculation. A policy such as `edge::precision::FP32`, or a user-defined precision policy, supplies the actual C++ types for parameters, activations, gradients, accumulators, optimizer state, and loss values. The planner uses `sizeof(T)` and `alignof(T)` for each of those types, then computes aligned offsets:

```cpp
static_assert(Model::parameter_count > 0);
static_assert(Model::required_memory >= Model::total_bytes);

constexpr auto memory = Model::memory_breakdown();
static_assert(memory.parameter_bytes == Model::parameter_bytes);
```

Conceptually the compile-time formula is:

```cpp
parameter_bytes  = parameter_count * sizeof(parameter_type);
gradient_bytes   = gradient_count * sizeof(gradient_type);
optimizer_bytes  = optimizer_state_count * sizeof(optimizer_state_type);
activation_bytes = activation_count * sizeof(activation_type);
workspace_count  = backward_slot0_count + backward_slot1_count + layer_workspace_count;
workspace_bytes  = workspace_count * sizeof(accumulator_type);

gradient_offset  = align_up(parameter_offset + parameter_bytes, alignof(gradient_type));
optimizer_offset = align_up(gradient_offset + gradient_bytes, alignof(optimizer_state_type));
activation_offset = align_up(optimizer_offset + optimizer_bytes, alignof(activation_type));
workspace_offset = align_up(activation_offset + activation_bytes, alignof(accumulator_type));

required_memory  = align_up(workspace_offset + workspace_bytes, Model::alignment);
```

`sizeof(T)` returns how many bytes one object of type `T` occupies. `alignof(T)` returns the address alignment that type requires. `align_up(...)` inserts padding between arena sections so each typed pointer is correctly aligned.

The important point is that no runtime descriptor is needed to discover how much memory the model needs. The runtime constructor only binds pointers into the already-sized arena and checks the address alignment.

This also means external arenas can be statically checked when their size is part of the type:

```cpp
alignas(Model::alignment)
static std::array<std::byte, Model::required_memory> arena{};

Model model{edge::external_arena(arena)};
static_assert(decltype(edge::external_arena(arena))::size == Model::required_memory);
```

The key is the `N` in `std::array<std::byte, N>`. Because `N` is part of the type, `external_arena(arena)` returns `ExternalArena<N>`, and the model constructor can compare `N` against `Model::required_memory` with `static_assert`.

If the buffer is too small, compilation fails:

```cpp
alignas(Model::alignment)
static std::array<std::byte, Model::required_memory - 1> too_small{};

// Model bad{edge::external_arena(too_small)};
// error: External arena is too small for this model; use Model::required_memory
```

Alignment is still checked at construction time because `std::array<std::byte, N>`, `std::span<std::byte, N>`, and C array references encode the byte count, but not a target-specific alignment guarantee for the object address. For embedded firmware, declare external storage with `alignas(Model::alignment)`.

The full flow is:

1. The user writes a model type with `edge::Input`, layer specs, optional backend policy, and optional precision policy.
2. Each layer spec becomes a layer instance for the current input size.
3. Each instance exposes compile-time traits: output size, parameter count, cache count, and workspace count.
4. `LayerChain` accumulates those traits across the network.
5. `ModelImpl` converts counts into bytes using the precision policy types.
6. `ModelImpl` computes aligned section offsets and `Model::required_memory`.
7. The internal arena uses `std::array<std::byte, Model::required_memory>`.
8. External arenas use `std::array`, fixed-extent `std::span`, or C array references so the constructor can reject undersized buffers at compile time.

This compile-time planner was introduced with the initial EdgeLearning++ C++20 runtime on 2026-06-23 in commit `533ce92`. It was extended on 2026-06-23 in commit `2fb0842` so precision policies and custom layers feed the same planner. The embedded storage guidance for static internal arenas and linker-placeable external arenas was documented on 2026-06-23 in commit `6d5b137`.

The planner also accounts for the model precision policy. For example, if `ParameterT` is `double`, `parameter_bytes` is `parameter_count * sizeof(double)`. If a policy keeps activations as `float` but accumulates in a wider type, `activation_bytes` and `workspace_bytes` are computed independently and aligned independently.

External arenas are accepted only through APIs where the byte count is part of the type:

```cpp
std::array<std::byte, N>&
std::span<std::byte, N>
std::byte (&)[N]
```

The constructor statically rejects `N < Model::required_memory`. Alignment is checked at runtime because variable alignment is not encoded in those types. Pass `alignas(Model::alignment)` storage.

This avoids common embedded failure modes: stack overflow from large local objects, runtime heap allocation failure, and shared RTOS heap fragmentation. The training path binds a statically sized arena once, checks status, and then runs without heap allocation. A `void* + size_t` API would not have the same compile-time property; if added later, it must return `Status` and be documented as runtime-checked only.

A future multi-region planner can split parameters, activations, workspace, and optimizer state across memory regions. In v0.1, users who need DTCM or another region should place the entire arena there.
