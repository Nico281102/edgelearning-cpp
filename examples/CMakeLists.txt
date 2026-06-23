set(EDGE_EXAMPLES
    simple_regression
    external_arena
    custom_loss
    custom_activation
    custom_precision
    custom_layer
    conv2d_mnist_shape
)

foreach(example_name IN LISTS EDGE_EXAMPLES)
    add_executable(${example_name} ${example_name}.cpp)
    target_link_libraries(${example_name} PRIVATE edge::edgelearning)
endforeach()
