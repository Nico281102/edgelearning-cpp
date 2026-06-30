set(EDGE_BENCHMARK_HOST_M2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/host/m2")
set(EDGE_BENCHMARK_FIRMWARE_STM32N6_DIR "${CMAKE_CURRENT_SOURCE_DIR}/firmware/stm32n6")

add_executable(benchmark_edgelearning_cpp
    host/m2/runtime/benchmark_edgelearning_cpp.cpp
)
target_link_libraries(benchmark_edgelearning_cpp PRIVATE edge::edgelearning)

add_executable(benchmark_batch_sweep
    host/m2/batch_sweep/benchmark_batch_sweep.cpp
)
target_link_libraries(benchmark_batch_sweep PRIVATE edge::edgelearning)

add_executable(benchmark_mixed_precision
    host/m2/mixed_precision/benchmark_mixed_precision.cpp
)
target_link_libraries(benchmark_mixed_precision PRIVATE edge::edgelearning)
target_compile_definitions(benchmark_mixed_precision PRIVATE
    EDGE_BENCHMARK_RESULT_DIR="${EDGE_BENCHMARK_HOST_M2_DIR}/mixed_precision/results"
)

add_executable(benchmark_regression_vs_c_baseline
    host/m2/c_baseline_regression/benchmark_regression_vs_c_baseline.cpp
)
target_link_libraries(benchmark_regression_vs_c_baseline PRIVATE edge::edgelearning)

add_executable(code_size_cpp_regression
    host/m2/c_baseline_regression/code_size_cpp_regression.cpp
)
target_link_libraries(code_size_cpp_regression PRIVATE edge::edgelearning)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(code_size_cpp_regression PRIVATE
        -Os
        -ffunction-sections
        -fdata-sections
    )
endif()
if(APPLE)
    target_link_options(code_size_cpp_regression PRIVATE -Wl,-dead_strip)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(code_size_cpp_regression PRIVATE -Wl,--gc-sections)
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_custom_target(benchmark_code_size
        COMMAND "${Python3_EXECUTABLE}"
                "${EDGE_BENCHMARK_HOST_M2_DIR}/c_baseline_regression/benchmark_code_size.py"
                --cpp-bin "$<TARGET_FILE:code_size_cpp_regression>"
                --result-dir "${EDGE_BENCHMARK_HOST_M2_DIR}/c_baseline_regression/results"
                --repo "${CMAKE_SOURCE_DIR}"
        DEPENDS code_size_cpp_regression
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Measuring EdgeLearning++ code size; set EDGE_C_BASELINE_DIR for old-C comparison"
        VERBATIM
    )
endif()

get_property(EDGE_M55_REGRESSION_ELF_TARGETS GLOBAL PROPERTY EDGE_M55_REGRESSION_ELF_TARGETS)
if(Python3_Interpreter_FOUND AND EDGE_M55_REGRESSION_ELF_TARGETS)
    set(EDGE_M55_REGRESSION_ELF_BINARY_ARGS)
    foreach(edge_m55_target IN LISTS EDGE_M55_REGRESSION_ELF_TARGETS)
        list(APPEND EDGE_M55_REGRESSION_ELF_BINARY_ARGS
            --binary "${edge_m55_target}=$<TARGET_FILE:${edge_m55_target}>"
        )
    endforeach()
    add_custom_target(m55_regression_elf_size
        COMMAND "${Python3_EXECUTABLE}"
                "${EDGE_BENCHMARK_FIRMWARE_STM32N6_DIR}/m55_regression/measure_m55_regression_elf.py"
                --result-dir "${EDGE_BENCHMARK_FIRMWARE_STM32N6_DIR}/m55_regression/results"
                ${EDGE_M55_REGRESSION_ELF_BINARY_ARGS}
        DEPENDS ${EDGE_M55_REGRESSION_ELF_TARGETS}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Measuring static regression sweep ELF section sizes"
        VERBATIM
    )
endif()
execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --short HEAD
    OUTPUT_VARIABLE EDGE_CPP_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}" status --short -- .
            ":!benchmarks/host/m2/batch_sweep/results"
            ":!benchmarks/host/m2/mixed_precision/results"
            ":!benchmarks/host/m2/c_baseline_regression/results"
            ":!benchmarks/firmware/stm32n6/m55_regression/results"
            ":!benchmarks/firmware/stm32n6/el_cvscpp_ablation/results"
    OUTPUT_VARIABLE EDGE_CPP_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT EDGE_CPP_GIT_COMMIT)
    set(EDGE_CPP_GIT_COMMIT "unknown")
endif()
if(EDGE_CPP_GIT_STATUS)
    string(APPEND EDGE_CPP_GIT_COMMIT " (dirty)")
endif()
target_compile_definitions(benchmark_regression_vs_c_baseline PRIVATE
    EDGE_BENCHMARK_RESULT_DIR="${EDGE_BENCHMARK_HOST_M2_DIR}/c_baseline_regression/results"
    EDGE_CPP_GIT_COMMIT="${EDGE_CPP_GIT_COMMIT}"
    EDGE_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    EDGE_CXX_STANDARD="C++20"
)
