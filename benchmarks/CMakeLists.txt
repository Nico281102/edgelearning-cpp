add_executable(benchmark_edgelearning_cpp benchmark_edgelearning_cpp.cpp)
target_link_libraries(benchmark_edgelearning_cpp PRIVATE edge::edgelearning)

add_executable(benchmark_batch_sweep benchmark_batch_sweep.cpp)
target_link_libraries(benchmark_batch_sweep PRIVATE edge::edgelearning)

add_executable(benchmark_regression_vs_c_baseline benchmark_regression_vs_c_baseline.cpp)
target_link_libraries(benchmark_regression_vs_c_baseline PRIVATE edge::edgelearning)
execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --short HEAD
    OUTPUT_VARIABLE EDGE_CPP_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}" status --short -- .
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
