cmake_minimum_required(VERSION 3.20)

project(edgelearning_cpp VERSION 0.1.0 LANGUAGES CXX)

option(EDGE_BUILD_TESTS "Build EdgeLearning++ host tests" ON)
option(EDGE_BUILD_BENCHMARKS "Build EdgeLearning++ host benchmarks" ON)
option(EDGE_BUILD_EXAMPLES "Build EdgeLearning++ examples" ON)

add_library(edgelearning_cpp INTERFACE)
add_library(edge::edgelearning ALIAS edgelearning_cpp)

target_compile_features(edgelearning_cpp INTERFACE cxx_std_20)
target_include_directories(edgelearning_cpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(edgelearning_cpp INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
    )
endif()

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(EDGE_MACOS_SDK "${CMAKE_OSX_SYSROOT}")
    if(NOT EDGE_MACOS_SDK OR NOT IS_ABSOLUTE "${EDGE_MACOS_SDK}")
        execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-path
            OUTPUT_VARIABLE EDGE_MACOS_SDK
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
    if(EXISTS "${EDGE_MACOS_SDK}/usr/include/c++/v1/array")
        target_include_directories(edgelearning_cpp SYSTEM INTERFACE
            "${EDGE_MACOS_SDK}/usr/include/c++/v1"
        )
    endif()
endif()

if(EDGE_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

if(EDGE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(EDGE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
