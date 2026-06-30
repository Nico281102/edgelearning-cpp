# Examples

Each example focuses on one API surface. They are built by CMake when
`EDGE_BUILD_EXAMPLES=ON`.

| Example | Focus |
|---|---|
| `minimal_regression.cpp` | One Adam training step on a Dense regression model |
| `minimal_regression_batch.cpp` | Effective batch size through gradient accumulation before one optimizer step |
| `external_arena_static.cpp` | Static external arena with `Model::required_memory` |
| `backend_policy.cpp` | Same topology with `Backend::Generic` and `Backend::M55` |
| `precision_policy.cpp` | Model-level arithmetic/storage type policy |
| `custom_layer_scale.cpp` | Shape-aware custom layer with trainable parameters |
| `custom_activation.cpp` | Custom activation policy |
| `custom_loss.cpp` | Custom loss with value-plus-gradient path |
| `conv2d_flatten_dense.cpp` | `CHW` input, Conv2D, Flatten, Dense |

Build all host examples:

```sh
cmake -S . -B build -DEDGE_BUILD_EXAMPLES=ON
cmake --build build --target minimal_regression minimal_regression_batch external_arena_static backend_policy
```

## Batch Size

Set the effective batch size on `edge::TrainerConfig`:

```cpp
edge::Trainer<Model, edge::MSE, edge::Adam> trainer(
    edge::TrainerConfig{
        .batch_size = 256,
        .reduction = edge::GradientReduction::Mean,
    },
    edge::AdamConfig{.learning_rate = 1.0e-3F});
```

`train_step(input, target)` still consumes one sample at a time. The trainer
accumulates gradients until `batch_size` samples have been seen, then applies
one optimizer step and clears the gradients. Call `flush()` at the end of an
epoch or stream to apply a final partial batch.
