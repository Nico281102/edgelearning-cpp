#include <array>

#include <edge/edge.hpp>

struct DoublePrecision {
    using ParameterT = double;
    using ActivationT = double;
    using GradientT = double;
    using AccumulatorT = double;
    using OptimizerStateT = double;
    using LossT = double;
};

int main() {
    using Model = edge::Model<
        DoublePrecision,
        edge::Input<2>,
        edge::Dense<4, edge::ReLU>,
        edge::Dense<1>>;

    static_assert(Model::parameter_bytes == Model::parameter_count * sizeof(double));
    static_assert(Model::required_memory > 0);

    edge::Trainer<Model, edge::MSE, edge::SGD> trainer(
        edge::SGDConfig{.learning_rate = 0.01F});
    trainer.model().initialize(edge::InitConfig{.seed = 4U});

    const std::array<double, 2> input{0.25, -0.75};
    const std::array<double, 1> target{0.5};
    return edge::is_ok(trainer.train_step(input, target)) ? 0 : 1;
}
