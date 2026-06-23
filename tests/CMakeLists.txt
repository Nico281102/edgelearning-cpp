set(EDGE_TESTS
    test_forward
    test_backward
    test_finite_difference
    test_trainer
    test_arena
    test_custom_loss
    test_custom_activation
    test_initializers
    test_optimizer
    test_serialization
    test_compile_time_contracts
    test_precision_policy
    test_custom_layer
)

foreach(test_name IN LISTS EDGE_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE edge::edgelearning)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
