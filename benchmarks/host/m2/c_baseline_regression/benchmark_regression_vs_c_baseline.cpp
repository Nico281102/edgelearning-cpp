#include <array>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include <edge/edge.hpp>

#ifndef EDGE_BENCHMARK_RESULT_DIR
#define EDGE_BENCHMARK_RESULT_DIR "benchmarks/host/m2/c_baseline_regression/results"
#endif

#ifndef EDGE_CPP_GIT_COMMIT
#define EDGE_CPP_GIT_COMMIT "unknown"
#endif

#ifndef EDGE_CXX_COMPILER
#define EDGE_CXX_COMPILER "unknown"
#endif

#ifndef EDGE_CXX_STANDARD
#define EDGE_CXX_STANDARD "C++20"
#endif

namespace {

using Model = edge::Model<
    edge::InputVector<8>,
    edge::Dense<32, edge::ReLU>,
    edge::Dense<16, edge::ReLU>,
    edge::Dense<1>>;

double run_cpp_timing(std::size_t iterations) {
    edge::Trainer<Model, edge::MSE, edge::Adam> trainer(
        edge::TrainerConfig{.batch_size = 8, .reduction = edge::GradientReduction::Mean},
        edge::AdamConfig{.learning_rate = 0.01F});
    trainer.model().initialize(edge::InitConfig{.seed = 99U});

    std::array<float, Model::input_size> input{};
    std::array<float, Model::output_size> target{0.5F};
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) * 0.01F;
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        trainer.train_step(input, target);
    }
    trainer.flush();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() /
           static_cast<double>(iterations);
}

std::string host_cpu_name() {
#if defined(__APPLE__)
    char buffer[256]{};
    std::size_t size = sizeof(buffer);
    if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, nullptr, 0) == 0 && buffer[0] != '\0') {
        return std::string(buffer);
    }
    size = sizeof(buffer);
    if (sysctlbyname("hw.model", buffer, &size, nullptr, 0) == 0 && buffer[0] != '\0') {
        return std::string(buffer);
    }
#endif
    return "unknown CPU";
}

void append_log_block(std::ostream& out,
                      const std::filesystem::path& log_path,
                      const std::string& begin_marker,
                      const std::string& end_marker) {
    std::ifstream log(log_path);
    if (!log) {
        return;
    }

    bool in_block = false;
    std::string line;
    while (std::getline(log, line)) {
        if (!in_block && line.find(begin_marker) != std::string::npos) {
            in_block = true;
        }
        if (in_block && !end_marker.empty() && line.find(end_marker) != std::string::npos) {
            break;
        }
        if (in_block) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            out << line << '\n';
        }
    }
}

std::filesystem::path baseline_log_path() {
    if (const char* path = std::getenv("EDGE_C_BASELINE_BENCHMARK_LOG")) {
        return path;
    }
    return std::filesystem::temp_directory_path() / "edgelearning_c_baseline_008581_benchmark.txt";
}

std::string display_baseline_log_path(const std::filesystem::path& path) {
    if (std::getenv("EDGE_C_BASELINE_BENCHMARK_LOG")) {
        return "${EDGE_C_BASELINE_BENCHMARK_LOG}";
    }
    if (path.filename() == "edgelearning_c_baseline_008581_benchmark.txt") {
        return "${TMPDIR}/edgelearning_c_baseline_008581_benchmark.txt";
    }
    return path.filename().string();
}

} // namespace

int main() {
    constexpr std::size_t iterations = 1000;
    const double cpp_train_step_ns = run_cpp_timing(iterations);

    std::filesystem::create_directories(EDGE_BENCHMARK_RESULT_DIR);
    const std::filesystem::path report_path =
        std::filesystem::path(EDGE_BENCHMARK_RESULT_DIR) / "host_regression_report.md";
    std::ofstream report(report_path);
    report << "# Host Regression Report\n\n";
    report << "Baseline C commit: `0085814908ca1b57ece4fe367361d084fd74aa3e`.\n\n";
    report << "C++ commit used: `" << EDGE_CPP_GIT_COMMIT << "`.\n\n";
    report << "This repository does not vendor or republish the old C source. The old C baseline is "
              "intended to be cloned or checked out locally outside `edgelearning-cpp` for side-by-side "
              "measurement using the same topology, seed, synthetic dataset, optimizer, and batch policy.\n\n";
    report << "## Host And Build\n\n";
    report << "- Compiler: " << EDGE_CXX_COMPILER << "\n";
    report << "- Language mode: " << EDGE_CXX_STANDARD << "\n";
    report << "- Core flags: `-fno-exceptions -fno-rtti`\n";
    report << "- Host CPU/model: " << host_cpu_name() << "\n";
    report << "- Hardware threads: " << std::thread::hardware_concurrency() << "\n\n";
    report << "## Current C++ Measurement\n\n";
    report << "- Topology: `InputVector<8>, Dense<32, ReLU>, Dense<16, ReLU>, Dense<1>`\n";
    report << "- Iterations: " << iterations << "\n";
    report << "- Optimizer: Adam, learning rate 0.01\n";
    report << "- Batch reduction: Mean, batch size 8\n";
    report << "- C++ train_step mean: " << cpp_train_step_ns << " ns\n";
    report << "- Parameter bytes: " << Model::parameter_bytes << "\n";
    report << "- Gradient bytes: " << Model::gradient_bytes << "\n";
    report << "- Optimizer bytes: " << Model::optimizer_bytes << "\n";
    report << "- Activation bytes: " << Model::activation_bytes << "\n";
    report << "- Workspace bytes: " << Model::workspace_bytes << "\n";
    report << "- Total planned bytes: " << Model::total_bytes << "\n\n";
    const std::filesystem::path code_size_report =
        std::filesystem::path(EDGE_BENCHMARK_RESULT_DIR) / "code_size_report.md";
    report << "## Code Size\n\n";
    if (std::filesystem::exists(code_size_report)) {
        report << "Code-size measurements are available in `"
               << code_size_report.filename().string()
               << "`. That report compares the linked C++ minimal training binary against the "
                  "old C baseline when `EDGE_C_BASELINE_DIR` was provided.\n\n";
    } else {
        report << "No code-size report was found yet. Build `code_size_cpp_regression`, then run:\n\n";
        report << "```sh\n";
        report << "EDGE_C_BASELINE_DIR=/path/to/EdgeLearning-at-0085814908ca1b57ece4fe367361d084fd74aa3e \\\n";
        report << "cmake --build build --target benchmark_code_size\n";
        report << "```\n\n";
    }
    const std::filesystem::path c_log = baseline_log_path();
    if (std::filesystem::exists(c_log)) {
        report << "## Old C Baseline Local Measurements\n\n";
        report << "Source checkout location used for this run: temporary directory outside this repository.\n\n";
        report << "Raw baseline benchmark log: `" << display_baseline_log_path(c_log)
               << "` (not committed to this repository).\n\n";
        report << "Selected result blocks:\n\n";
        report << "```text\n";
        append_log_block(report,
                         c_log,
                         "=== DATASET: simple_regression ===",
                         "=== DATASET: multi_regression ===");
        append_log_block(report,
                         c_log,
                         "=== DATASET: multi_regression ===",
                         "=== DATASET: deep_teacher_regression_p1213_l3 ===");
        report << "```\n\n";
    } else {
        report << "## Old C Baseline Local Measurements\n\n";
        report << "No old-C benchmark log was found at `" << display_baseline_log_path(c_log)
               << "`. Set `EDGE_C_BASELINE_BENCHMARK_LOG` to a local benchmark output file to include "
                  "measured C baseline rows in the generated report.\n\n";
    }
    report << "## C Baseline Methodology\n\n";
    report << "1. Check out `Nico281102/EdgeLearning` at the baseline commit outside this repository.\n";
    report << "2. Build the old C library with the same compiler family and optimization flags.\n";
    report << "3. Use the same flat parameter layout: layer order, row-major weights, then bias.\n";
    report << "4. Run the same synthetic samples for the same iteration count and report mean timing.\n";
    report << "5. Publish only measurements and methodology, not the old C source.\n";

    std::cout << "Wrote " << report_path << '\n';
    std::cout << "cpp_train_step_ns=" << cpp_train_step_ns << '\n';
}
