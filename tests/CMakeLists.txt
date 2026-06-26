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
    test_conv2d
    test_conv2d_finite_difference
    test_conv2d_mnist_shape
    test_backend_m55
)

function(edge_apply_test_optimization target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Ofast>
            $<$<COMPILE_LANGUAGE:C>:-Ofast>
        )
    endif()
endfunction()

foreach(test_name IN LISTS EDGE_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE edge::edgelearning)
    edge_apply_test_optimization(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

set(EDGE_C_BASELINE_DIR "" CACHE PATH
    "Optional local checkout of the legacy EdgeLearning C library; sources are linked from this path and not copied.")
set(EDGE_LEGACY_C_BACKEND "generic" CACHE STRING
    "Legacy C backend source to link for optional regression sweep targets: generic or m55.")
set_property(CACHE EDGE_LEGACY_C_BACKEND PROPERTY STRINGS generic m55)
set(EDGE_LEGACY_C_INCLUDE_DIRS "" CACHE STRING
    "Optional semicolon-separated include directories required by the selected legacy C backend.")

function(edge_add_m55_regression_case suffix hidden1 hidden2)
    set(target "test_m55_regression_${suffix}")
    add_executable(${target} test_m55_regression_sweep.cpp)
    target_compile_definitions(${target} PRIVATE
        EDGE_M55_SWEEP_H1=${hidden1}
        EDGE_M55_SWEEP_H2=${hidden2}
    )
    target_link_libraries(${target} PRIVATE edge::edgelearning)
    edge_apply_test_optimization(${target})
    add_test(NAME ${target} COMMAND ${target})
    set_property(GLOBAL APPEND PROPERTY EDGE_M55_REGRESSION_ELF_TARGETS ${target})
endfunction()

edge_add_m55_regression_case("8x8" 8 8)
edge_add_m55_regression_case("16x8" 16 8)
edge_add_m55_regression_case("16x16" 16 16)
edge_add_m55_regression_case("32x16" 32 16)
edge_add_m55_regression_case("32x32" 32 32)
edge_add_m55_regression_case("32x64" 32 64)
edge_add_m55_regression_case("64x64" 64 64)
edge_add_m55_regression_case("128x64" 128 64)

function(edge_legacy_c_sources out_var)
    set(legacy_root "${EDGE_C_BASELINE_DIR}")
    set(sources
        "${legacy_root}/0_Common/el_config.c"
        "${legacy_root}/0_Common/el_system.c"
        "${legacy_root}/1_Engine/el_model.c"
        "${legacy_root}/2_Network/el_layer.c"
        "${legacy_root}/2_Network/el_layer_factory.c"
        "${legacy_root}/2_Network/el_network.c"
        "${legacy_root}/2_Network/el_network_factory.c"
        "${legacy_root}/2_Network/el_network_params.c"
    )
    if(EDGE_LEGACY_C_BACKEND STREQUAL "m55")
        list(APPEND sources "${legacy_root}/1_Engine/el_backend_m55.c")
    elseif(EDGE_LEGACY_C_BACKEND STREQUAL "generic")
        list(APPEND sources "${legacy_root}/1_Engine/el_backend_general_cpu.c")
    else()
        message(FATAL_ERROR "EDGE_LEGACY_C_BACKEND must be 'generic' or 'm55'")
    endif()
    set(${out_var} ${sources} PARENT_SCOPE)
endfunction()

function(edge_add_legacy_c_regression_case kind suffix hidden1 hidden2)
    edge_legacy_c_sources(legacy_sources)
    set(target "test_legacy_c_${EDGE_LEGACY_C_BACKEND}_${kind}_regression_${suffix}")
    add_executable(${target}
        test_m55_regression_sweep.cpp
        ${legacy_sources}
    )
    if(kind STREQUAL "direct_backend")
        set(mode_definitions
            EDGE_M55_SWEEP_ENABLE_CPP_GENERIC=0
            EDGE_M55_SWEEP_ENABLE_CPP_M55=0
            EDGE_M55_SWEEP_ENABLE_DIRECT_C_BACKEND=1
        )
    elseif(kind STREQUAL "native_m55")
        set(mode_definitions
            EDGE_M55_SWEEP_ENABLE_CPP_GENERIC=0
            EDGE_M55_SWEEP_ENABLE_CPP_M55=1
            EDGE_M55_SWEEP_ENABLE_DIRECT_C_BACKEND=0
        )
    elseif(kind STREQUAL "generic_cpp")
        set(mode_definitions
            EDGE_M55_SWEEP_ENABLE_CPP_GENERIC=1
            EDGE_M55_SWEEP_ENABLE_CPP_M55=0
            EDGE_M55_SWEEP_ENABLE_DIRECT_C_BACKEND=0
        )
    else()
        message(FATAL_ERROR "unknown legacy C regression kind: ${kind}")
    endif()
    target_compile_definitions(${target} PRIVATE
        EDGE_M55_SWEEP_H1=${hidden1}
        EDGE_M55_SWEEP_H2=${hidden2}
        EDGE_M55_SWEEP_ENABLE_LEGACY_C=1
        ${mode_definitions}
    )
    target_include_directories(${target} PRIVATE
        "${EDGE_C_BASELINE_DIR}"
        "${EDGE_C_BASELINE_DIR}/Inc"
        ${EDGE_LEGACY_C_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PRIVATE edge::edgelearning)
    if(UNIX)
        target_link_libraries(${target} PRIVATE m)
    endif()
    edge_apply_test_optimization(${target})
    add_test(NAME ${target} COMMAND ${target})
    set_property(GLOBAL APPEND PROPERTY EDGE_M55_REGRESSION_ELF_TARGETS ${target})
endfunction()

if(EDGE_C_BASELINE_DIR)
    if(NOT EXISTS "${EDGE_C_BASELINE_DIR}/Inc/edgelearning.h")
        message(FATAL_ERROR
            "EDGE_C_BASELINE_DIR must point to a legacy EdgeLearning checkout containing Inc/edgelearning.h")
    endif()
    enable_language(C)
    foreach(edge_legacy_kind IN ITEMS direct_backend native_m55 generic_cpp)
        edge_add_legacy_c_regression_case(${edge_legacy_kind} "8x8" 8 8)
        edge_add_legacy_c_regression_case(${edge_legacy_kind} "16x8" 16 8)
        edge_add_legacy_c_regression_case(${edge_legacy_kind} "16x16" 16 16)
        edge_add_legacy_c_regression_case(${edge_legacy_kind} "32x16" 32 16)
        edge_add_legacy_c_regression_case(${edge_legacy_kind} "32x32" 32 32)
        edge_add_legacy_c_regression_case(${edge_legacy_kind} "32x64" 32 64)
        edge_add_legacy_c_regression_case(${edge_legacy_kind} "64x64" 64 64)
        edge_add_legacy_c_regression_case(${edge_legacy_kind} "128x64" 128 64)
    endforeach()
endif()
