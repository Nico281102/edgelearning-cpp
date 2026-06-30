set(EDGE_EXAMPLES
    minimal_regression
    minimal_regression_batch
    external_arena_static
    backend_policy
    custom_loss
    custom_activation
    precision_policy
    custom_layer_scale
    conv2d_flatten_dense
)

foreach(example_name IN LISTS EDGE_EXAMPLES)
    add_executable(${example_name} ${example_name}.cpp)
    target_link_libraries(${example_name} PRIVATE edge::edgelearning)
endforeach()
