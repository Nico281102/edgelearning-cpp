#include <array>
#include <chrono>
#include <cstddef>
#include <iostream>

#include <edge/edge.hpp>

namespace {

template<typename Model>
void fill_input(std::array<float, Model::input_size>& input) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        const int centered = static_cast<int>(i % 11U) - 5;
        input[i] = static_cast<float>(centered) * 0.05F;
    }
}

template<typename Model>
void fill_target(std::array<float, Model::output_size>& target) {
    for (std::size_t i = 0; i < target.size(); ++i) {
        target[i] = static_cast<float>(i + 1U) * 0.01F;
    }
}

template<typename Model>
void run_case(const char* name, std::size_t iterations) {
    Model model;
    model.initialize(edge::InitConfig{.seed = 123U});
    std::array<float, Model::input_size> input{};
    std::array<float, Model::output_size> target{};
    std::array<float, Model::output_size> output_gradient{};
    fill_input<Model>(input);
    fill_target<Model>(target);

    auto forward_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        model.forward(input);
    }
    auto forward_end = std::chrono::steady_clock::now();

    edge::TensorView<float, Model::output_size> output_gradient_view(output_gradient);
    edge::MSE::evaluate(model.output(), edge::TensorView<const float, Model::output_size>(target),
                        output_gradient_view);

    auto backward_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        model.zero_grad();
        model.backward(output_gradient);
    }
    auto backward_end = std::chrono::steady_clock::now();

    edge::Trainer<Model, edge::MSE, edge::Adam> trainer(
        edge::TrainerConfig{.batch_size = 1, .reduction = edge::GradientReduction::Mean},
        edge::AdamConfig{.learning_rate = 1.0e-3F});
    trainer.model().initialize(edge::InitConfig{.seed = 123U});

    auto train_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        trainer.train_step(input, target);
    }
    auto train_end = std::chrono::steady_clock::now();

    auto zero_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        model.zero_grad();
    }
    auto zero_end = std::chrono::steady_clock::now();

    edge::Adam adam;
    auto opt_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        adam.step(model, 1.0F);
    }
    auto opt_end = std::chrono::steady_clock::now();

    const auto ns = [](auto begin, auto end, std::size_t n) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() /
               static_cast<double>(n);
    };

    std::cout << name << ','
              << Model::input_size << ','
              << Model::output_size << ','
              << Model::parameter_bytes << ','
              << Model::gradient_bytes << ','
              << Model::optimizer_bytes << ','
              << Model::activation_bytes << ','
              << Model::workspace_bytes << ','
              << Model::total_bytes << ','
              << ns(forward_start, forward_end, iterations) << ','
              << ns(backward_start, backward_end, iterations) << ','
              << ns(train_start, train_end, iterations) << ','
              << ns(opt_start, opt_end, iterations) << ','
              << ns(zero_start, zero_end, iterations) << '\n';
}

} // namespace

int main() {
    constexpr std::size_t iterations = 1000;
    std::cout << "case,input,output,param_bytes,grad_bytes,opt_bytes,act_bytes,workspace_bytes,total_bytes,"
                 "forward_ns,backward_ns,train_step_ns,optimizer_ns,zero_grad_ns\n";
    run_case<edge::Model<edge::InputVector<8>, edge::Dense<16, edge::ReLU>, edge::Dense<1>>>(
        "8-16-1", iterations);
    run_case<edge::Model<
        edge::InputVector<8>, edge::Dense<32, edge::ReLU>, edge::Dense<16, edge::ReLU>, edge::Dense<1>>>(
        "8-32-16-1", iterations);
    run_case<edge::Model<
        edge::InputVector<32>, edge::Dense<64, edge::ReLU>, edge::Dense<32, edge::ReLU>, edge::Dense<4>>>(
        "32-64-32-4", iterations);
    run_case<edge::Model<
        edge::InputVector<128>, edge::Dense<64, edge::ReLU>, edge::Dense<32, edge::ReLU>, edge::Dense<3>>>(
        "128-64-32-3", iterations);
}
