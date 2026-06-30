#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <edge/edge.hpp>

#ifndef EDGE_BENCHMARK_RESULT_DIR
#define EDGE_BENCHMARK_RESULT_DIR "benchmarks/host/m2/mixed_precision/results"
#endif

namespace {

constexpr std::size_t kInputSize = 16;
constexpr std::size_t kSamples = 128;
constexpr std::size_t kEpochs = 120;
constexpr std::size_t kRepeats = 5;

using FP32Model = edge::Model<
    edge::precision::FP32,
    edge::InputVector<kInputSize>,
    edge::Dense<32, edge::Tanh>,
    edge::Dense<16, edge::Tanh>,
    edge::Dense<1>>;

using MixedModel = edge::Model<
    edge::precision::MixedFP16,
    edge::InputVector<kInputSize>,
    edge::Dense<32, edge::Tanh>,
    edge::Dense<16, edge::Tanh>,
    edge::Dense<1>>;

template<typename T>
T cast_activation(float value) {
    return static_cast<T>(value);
}

float feature_value(std::size_t sample, std::size_t feature) {
    const float s = static_cast<float>(sample + 1U);
    const float f = static_cast<float>(feature + 1U);
    return 0.6F * std::sin(0.031F * s * f) + 0.4F * std::cos(0.017F * (s + f));
}

float target_value(const std::array<float, kInputSize>& x) {
    return 0.35F * x[0] - 0.25F * x[3] + 0.15F * x[5] * x[5] -
           0.10F * x[7] + 0.05F * std::sin(x[1] + x[2]);
}

template<typename Model>
struct Dataset {
    using ActivationT = typename Model::activation_type;
    std::array<std::array<ActivationT, Model::input_size>, kSamples> inputs{};
    std::array<std::array<ActivationT, Model::output_size>, kSamples> targets{};
};

template<typename Model>
Dataset<Model> make_dataset() {
    Dataset<Model> dataset{};
    for (std::size_t sample = 0; sample < kSamples; ++sample) {
        std::array<float, kInputSize> fp32_input{};
        for (std::size_t feature = 0; feature < kInputSize; ++feature) {
            fp32_input[feature] = feature_value(sample, feature);
            dataset.inputs[sample][feature] =
                cast_activation<typename Model::activation_type>(fp32_input[feature]);
        }
        dataset.targets[sample][0] =
            cast_activation<typename Model::activation_type>(target_value(fp32_input));
    }
    return dataset;
}

template<typename Model>
float evaluate_loss(edge::Trainer<Model, edge::MSE, edge::Adam>& trainer,
                    const Dataset<Model>& dataset) {
    using LossT = typename Model::loss_type;
    LossT sum = LossT{0};
    for (std::size_t sample = 0; sample < kSamples; ++sample) {
        trainer.model().forward(dataset.inputs[sample]);
        sum += static_cast<LossT>(
            edge::MSE::value(trainer.model().output(),
                             edge::TensorView<const typename Model::activation_type,
                                              Model::output_size>(dataset.targets[sample])));
    }
    return static_cast<float>(sum / static_cast<LossT>(kSamples));
}

struct RunResult {
    std::vector<float> epoch_losses;
    std::vector<double> train_step_ns;
    float final_loss = 0.0F;
};

template<typename Model>
RunResult run_training() {
    const auto dataset = make_dataset<Model>();
    RunResult result{};
    result.epoch_losses.reserve(kEpochs);

    {
        edge::Trainer<Model, edge::MSE, edge::Adam> trainer(
            edge::TrainerConfig{.batch_size = 1, .reduction = edge::GradientReduction::Mean},
            edge::AdamConfig{.learning_rate = 5.0e-4F});
        trainer.model().initialize(edge::InitConfig{.seed = 42U});

        for (std::size_t epoch = 0; epoch < kEpochs; ++epoch) {
            for (std::size_t sample = 0; sample < kSamples; ++sample) {
                trainer.train_step(dataset.inputs[sample], dataset.targets[sample]);
            }
            result.epoch_losses.push_back(evaluate_loss(trainer, dataset));
        }
        result.final_loss = result.epoch_losses.back();
    }

    for (std::size_t repeat = 0; repeat < kRepeats; ++repeat) {
        edge::Trainer<Model, edge::MSE, edge::Adam> trainer(
            edge::TrainerConfig{.batch_size = 1, .reduction = edge::GradientReduction::Mean},
            edge::AdamConfig{.learning_rate = 5.0e-4F});
        trainer.model().initialize(edge::InitConfig{.seed = 42U});

        const auto start = std::chrono::steady_clock::now();
        for (std::size_t epoch = 0; epoch < kEpochs; ++epoch) {
            for (std::size_t sample = 0; sample < kSamples; ++sample) {
                trainer.train_step(dataset.inputs[sample], dataset.targets[sample]);
            }
        }
        const auto end = std::chrono::steady_clock::now();
        const auto elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        result.train_step_ns.push_back(
            static_cast<double>(elapsed_ns) / static_cast<double>(kEpochs * kSamples));
    }

    return result;
}

double mean(std::vector<double> values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

double min_value(const std::vector<double>& values) {
    return *std::min_element(values.begin(), values.end());
}

template<typename Model>
void write_summary_row(std::ofstream& out,
                       const char* policy,
                       bool native_fp16,
                       const RunResult& result) {
    out << policy << ','
        << (native_fp16 ? "true" : "false") << ','
        << Model::parameter_bytes << ','
        << Model::gradient_bytes << ','
        << Model::optimizer_bytes << ','
        << Model::activation_bytes << ','
        << Model::workspace_bytes << ','
        << Model::required_memory << ','
        << result.final_loss << ','
        << mean(result.train_step_ns) << ','
        << median(result.train_step_ns) << ','
        << min_value(result.train_step_ns) << ','
        << kEpochs << ','
        << kSamples << ','
        << kRepeats << '\n';
}

void write_convergence_csv(const std::filesystem::path& path,
                           const RunResult& fp32,
                           const RunResult& mixed) {
    std::ofstream out(path);
    out << "epoch,fp32_loss,mixed_fp16_loss\n";
    for (std::size_t i = 0; i < fp32.epoch_losses.size(); ++i) {
        out << (i + 1U) << ','
            << fp32.epoch_losses[i] << ','
            << mixed.epoch_losses[i] << '\n';
    }
}

void write_svg(const std::filesystem::path& path,
               const RunResult& fp32,
               const RunResult& mixed) {
    constexpr int width = 820;
    constexpr int height = 420;
    constexpr int left = 64;
    constexpr int right = 24;
    constexpr int top = 28;
    constexpr int bottom = 52;
    const int plot_w = width - left - right;
    const int plot_h = height - top - bottom;

    std::vector<float> all = fp32.epoch_losses;
    all.insert(all.end(), mixed.epoch_losses.begin(), mixed.epoch_losses.end());
    const float min_loss = *std::min_element(all.begin(), all.end());
    const float max_loss = *std::max_element(all.begin(), all.end());
    const float range = std::max(max_loss - min_loss, 1.0e-6F);

    const auto point = [&](std::size_t index, float loss) {
        const double x = static_cast<double>(left) +
                         static_cast<double>(index) * static_cast<double>(plot_w) /
                             static_cast<double>(kEpochs - 1U);
        const double y = static_cast<double>(top + plot_h) -
                         static_cast<double>(loss - min_loss) * static_cast<double>(plot_h) /
                             static_cast<double>(range);
        return std::pair<double, double>{x, y};
    };

    const auto polyline = [&](const std::vector<float>& losses) {
        std::string points;
        for (std::size_t i = 0; i < losses.size(); ++i) {
            const auto [x, y] = point(i, losses[i]);
            points += std::to_string(x) + "," + std::to_string(y) + " ";
        }
        return points;
    };

    std::ofstream out(path);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << ' ' << height
        << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n";
    out << "<text x=\"24\" y=\"22\" font-family=\"Arial\" font-size=\"16\" "
           "fill=\"#111111\">FP32 vs Mixed FP16 convergence, MacBook M2</text>\n";
    out << "<line x1=\"" << left << "\" y1=\"" << top << "\" x2=\"" << left
        << "\" y2=\"" << (top + plot_h) << "\" stroke=\"#333333\"/>\n";
    out << "<line x1=\"" << left << "\" y1=\"" << (top + plot_h) << "\" x2=\""
        << (left + plot_w) << "\" y2=\"" << (top + plot_h)
        << "\" stroke=\"#333333\"/>\n";
    out << "<polyline fill=\"none\" stroke=\"#0b66c3\" stroke-width=\"2.5\" points=\""
        << polyline(fp32.epoch_losses) << "\"/>\n";
    out << "<polyline fill=\"none\" stroke=\"#c23b22\" stroke-width=\"2.5\" points=\""
        << polyline(mixed.epoch_losses) << "\"/>\n";
    out << "<text x=\"" << (left + plot_w - 170) << "\" y=\"54\" font-family=\"Arial\" "
           "font-size=\"13\" fill=\"#0b66c3\">FP32 baseline</text>\n";
    out << "<text x=\"" << (left + plot_w - 170) << "\" y=\"74\" font-family=\"Arial\" "
           "font-size=\"13\" fill=\"#c23b22\">MixedFP16 activations</text>\n";
    out << "<text x=\"" << left << "\" y=\"" << (height - 16) << "\" font-family=\"Arial\" "
           "font-size=\"12\" fill=\"#333333\">epoch</text>\n";
    out << "<text x=\"12\" y=\"" << (top + 12) << "\" font-family=\"Arial\" "
           "font-size=\"12\" fill=\"#333333\">MSE</text>\n";
    out << "<text x=\"" << (left + plot_w - 70) << "\" y=\"" << (height - 16)
        << "\" font-family=\"Arial\" font-size=\"12\" fill=\"#333333\">" << kEpochs
        << " epochs</text>\n";
    out << "<text x=\"12\" y=\"" << (top + plot_h + 4)
        << "\" font-family=\"Arial\" font-size=\"11\" fill=\"#666666\">" << min_loss
        << "</text>\n";
    out << "<text x=\"12\" y=\"" << (top + 4)
        << "\" font-family=\"Arial\" font-size=\"11\" fill=\"#666666\">" << max_loss
        << "</text>\n";
    out << "</svg>\n";
}

template<typename Model>
std::size_t activation_saving_percent() {
    if constexpr (std::is_same_v<Model, MixedModel>) {
        return 100U - (100U * MixedModel::activation_bytes / FP32Model::activation_bytes);
    }
    return 0U;
}

void write_markdown(const std::filesystem::path& path,
                    const RunResult& fp32,
                    const RunResult& mixed) {
    std::ofstream out(path);
    out << "# Mixed Precision Benchmark\n\n";
    out << "Host: MacBook M2, Darwin arm64. Build: C++20 Release. "
           "Generated by `benchmark_mixed_precision`.\n\n";
    out << "Policy: FP32 baseline stores parameters, activations, gradients, accumulators, "
           "optimizer state, and loss values as `float`. `MixedFP16` follows the common "
           "mixed-precision pattern of FP32 master parameters/gradients/accumulators with "
           "FP16 activation storage when `_Float16` is available. This is comparable in "
           "spirit to FP16 mixed precision training described by Micikevicius et al.; this "
           "small host benchmark does not use dynamic loss scaling because gradients and "
           "parameters remain FP32.\n\n";
    out << "Topology: `InputVector<16> -> Dense<32,Tanh> -> Dense<16,Tanh> -> Dense<1,Linear>`. "
        << "Dataset: deterministic synthetic regression, " << kSamples << " samples, "
        << kEpochs << " epochs, Adam, batch size 1. Timing reports mean/median/min "
        << "train-step nanoseconds over " << kRepeats << " fresh training runs.\n\n";
    out << "![Mixed precision convergence](mixed_precision_convergence.svg)\n\n";
    out << "| Policy | Native FP16 activation | Activation bytes | Required memory | Final MSE | Mean train step ns | Median train step ns |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|\n";
    out << "| FP32 | false | " << FP32Model::activation_bytes << " | "
        << FP32Model::required_memory << " | " << fp32.final_loss << " | "
        << mean(fp32.train_step_ns) << " | " << median(fp32.train_step_ns) << " |\n";
    out << "| MixedFP16 | "
        << (edge::precision::MixedFP16::uses_native_float16_storage ? "true" : "false")
        << " | " << MixedModel::activation_bytes << " | " << MixedModel::required_memory
        << " | " << mixed.final_loss << " | " << mean(mixed.train_step_ns)
        << " | " << median(mixed.train_step_ns) << " |\n\n";
    out << "Activation memory saving in this topology: "
        << activation_saving_percent<MixedModel>() << "%.\n\n";
    out << "Interpretation: this host result is a convergence and API smoke test, not a "
           "claim that FP16 is faster on every CPU. On MacBook M2, FP16 activation storage "
           "can reduce arena activation bytes, while runtime may be limited by conversion "
           "costs and scalar CPU code. MCU or accelerator backends must be measured with "
           "their target compiler and kernels.\n";
}

} // namespace

int main() {
    const auto result_dir = std::filesystem::path(EDGE_BENCHMARK_RESULT_DIR);
    std::filesystem::create_directories(result_dir);

    const RunResult fp32 = run_training<FP32Model>();
    const RunResult mixed = run_training<MixedModel>();

    {
        std::ofstream summary(result_dir / "mixed_precision_summary.csv");
        summary << "policy,native_fp16_activation,parameter_bytes,gradient_bytes,optimizer_bytes,"
                   "activation_bytes,workspace_bytes,required_memory,final_loss,"
                   "mean_train_step_ns,median_train_step_ns,min_train_step_ns,epochs,samples,repeats\n";
        write_summary_row<FP32Model>(summary, "FP32", false, fp32);
        write_summary_row<MixedModel>(
            summary,
            "MixedFP16",
            edge::precision::MixedFP16::uses_native_float16_storage,
            mixed);
    }

    write_convergence_csv(result_dir / "mixed_precision_convergence.csv", fp32, mixed);
    write_svg(result_dir / "mixed_precision_convergence.svg", fp32, mixed);
    write_markdown(result_dir / "mixed_precision_report.md", fp32, mixed);
}
