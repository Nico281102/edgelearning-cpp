add_executable(benchmark_edgelearning_cpp benchmark_edgelearning_cpp.cpp)
target_link_libraries(benchmark_edgelearning_cpp PRIVATE edge::edgelearning)

add_executable(benchmark_batch_sweep benchmark_batch_sweep.cpp)
target_link_libraries(benchmark_batch_sweep PRIVATE edge::edgelearning)

add_executable(benchmark_regression_vs_c_baseline benchmark_regression_vs_c_baseline.cpp)
target_link_libraries(benchmark_regression_vs_c_baseline PRIVATE edge::edgelearning)

add_executable(code_size_cpp_regression code_size_cpp_regression.cpp)
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
                "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_code_size.py"
                --cpp-bin "$<TARGET_FILE:code_size_cpp_regression>"
                --result-dir "${CMAKE_CURRENT_SOURCE_DIR}/results"
                --repo "${CMAKE_SOURCE_DIR}"
        DEPENDS code_size_cpp_regression
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Measuring EdgeLearning++ code size; set EDGE_C_BASELINE_DIR for old-C comparison"
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
    COMMAND git -C "${CMAKE_SOURCE_DIR}" status --short -- . ":!benchmarks/results"
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
    EDGE_BENCHMARK_RESULT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/results"
    EDGE_CPP_GIT_COMMIT="${EDGE_CPP_GIT_COMMIT}"
    EDGE_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    EDGE_CXX_STANDARD="C++20"
)
