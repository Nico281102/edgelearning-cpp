#include <array>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include <edge/edge.hpp>

namespace {

template<typename Model>
void fill_sample(std::array<float, Model::input_size>& input,
                 std::array<float, Model::output_size>& target,
                 std::size_t sample_index) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        const int centered = static_cast<int>((i + sample_index) % 17U) - 8;
        input[i] = static_cast<float>(centered) * 0.03125F;
    }
    for (std::size_t i = 0; i < target.size(); ++i) {
        const int centered = static_cast<int>((i + sample_index) % 7U) - 3;
        target[i] = static_cast<float>(centered) * 0.05F;
    }
}

template<typename Model>
double run_train(std::size_t batch_size, std::size_t samples) {
    edge::Trainer<Model, edge::MSE, edge::Adam> trainer(
        edge::TrainerConfig{.batch_size = batch_size, .reduction = edge::GradientReduction::Mean},
        edge::AdamConfig{.learning_rate = 1.0e-3F});
    trainer.model().initialize(edge::InitConfig{.seed = 123U});

    std::array<float, Model::input_size> input{};
    std::array<float, Model::output_size> target{};

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < samples; ++i) {
        fill_sample<Model>(input, target, i);
        trainer.train_step(input, target);
    }
    trainer.flush();
    const auto end = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() /
           static_cast<double>(samples);
}

template<typename Model>
double run_train(std::size_t batch_size, std::size_t samples, std::uint32_t seed) {
    edge::Trainer<Model, edge::MSE, edge::Adam> trainer(
        edge::TrainerConfig{.batch_size = batch_size, .reduction = edge::GradientReduction::Mean},
        edge::AdamConfig{.learning_rate = 1.0e-3F});
    trainer.model().initialize(edge::InitConfig{.seed = seed});

    std::array<float, Model::input_size> input{};
    std::array<float, Model::output_size> target{};

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < samples; ++i) {
        fill_sample<Model>(input, target, i + static_cast<std::size_t>(seed));
        trainer.train_step(input, target);
    }
    trainer.flush();
    const auto end = std::chrono::steady_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() /
           static_cast<double>(samples);
}

template<typename Model>
void run_case(const char* name, std::size_t samples) {
    constexpr std::array<std::size_t, 7> batches{1, 8, 16, 32, 64, 128, 256};
    constexpr std::array<std::uint32_t, 7> seeds{1U, 7U, 17U, 42U, 99U, 123U, 2027U};
    for (std::size_t batch : batches) {
        std::array<double, seeds.size()> timings{};
        double sum = 0.0;
        for (std::size_t i = 0; i < seeds.size(); ++i) {
            timings[i] = run_train<Model>(batch, samples, seeds[i]);
            sum += timings[i];
        }
        std::sort(timings.begin(), timings.end());
        std::cout << name << ','
                  << Model::input_size << ','
                  << Model::output_size << ','
                  << Model::parameter_count << ','
                  << Model::total_bytes << ','
                  << batch << ','
                  << samples << ','
                  << seeds.size() << ','
                  << timings.front() << ','
                  << timings[timings.size() / 2U] << ','
                  << (sum / static_cast<double>(seeds.size())) << ','
                  << timings.back() << '\n';
    }
}

} // namespace

int main() {
    constexpr std::size_t samples = 4096;
    std::cout << "case,input,output,parameter_count,total_bytes,batch,samples,seeds,min_ns,median_ns,mean_ns,max_ns\n";

    run_case<edge::Model<
        edge::InputVector<8>,
        edge::Dense<32, edge::ReLU>,
        edge::Dense<16, edge::ReLU>,
        edge::Dense<1>>>("8-32-16-1", samples);

    run_case<edge::Model<
        edge::InputVector<8>,
        edge::Dense<64, edge::ReLU>,
        edge::Dense<64, edge::ReLU>,
        edge::Dense<1>>>("8-64-64-1", samples);

    run_case<edge::Model<
        edge::InputVector<8>,
        edge::Dense<32, edge::ReLU>,
        edge::Dense<52, edge::ReLU>,
        edge::Dense<52, edge::ReLU>,
        edge::Dense<1>>>("8-32-52-52-1", samples);

    run_case<edge::Model<
        edge::InputVector<8>,
        edge::Dense<20, edge::ReLU>,
        edge::Dense<36, edge::ReLU>,
        edge::Dense<52, edge::ReLU>,
        edge::Dense<36, edge::ReLU>,
        edge::Dense<1>>>("8-20-36-52-36-1", samples);

    run_case<edge::Model<
        edge::InputVector<32>,
        edge::Dense<128, edge::ReLU>,
        edge::Dense<64, edge::ReLU>,
        edge::Dense<16>>>("32-128-64-16", samples);

    run_case<edge::Model<
        edge::InputVector<128>,
        edge::Dense<128, edge::ReLU>,
        edge::Dense<64, edge::ReLU>,
        edge::Dense<3>>>("128-128-64-3", samples);

    run_case<edge::Model<
        edge::InputVector<256>,
        edge::Dense<128, edge::ReLU>,
        edge::Dense<64, edge::ReLU>,
        edge::Dense<32, edge::ReLU>,
        edge::Dense<8>>>("256-128-64-32-8", samples);

    run_case<edge::Model<
        edge::InputVector<512>,
        edge::Dense<256, edge::ReLU>,
        edge::Dense<128, edge::ReLU>,
        edge::Dense<32, edge::ReLU>,
        edge::Dense<4>>>("512-256-128-32-4", samples);
}
