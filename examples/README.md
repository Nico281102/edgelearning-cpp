# Examples

Each example focuses on one API surface. They are built by CMake when
`EDGE_BUILD_EXAMPLES=ON`.

| Example | Focus |
|---|---|
| `minimal_regression.cpp` | One Adam training step on a Dense regression model |
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
cmake --build build --target minimal_regression external_arena_static backend_policy
```
