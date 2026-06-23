#include <array>
#include <cstddef>
#include <type_traits>

#include <edge/edge.hpp>

#include "test_harness.hpp"

namespace {

struct DoublePrecision {
    using ParameterT = double;
    using ActivationT = double;
    using GradientT = double;
    using AccumulatorT = double;
    using OptimizerStateT = double;
    using LossT = double;
};

} // namespace

int main() {
    static_assert(edge::PrecisionPolicy<DoublePrecision>);

    using Model = edge::Model<
        edge::Backend::Generic,
        DoublePrecision,
        edge::Input<2>,
        edge::Dense<1, edge::Linear>>;

    static_assert(std::is_same_v<Model::parameter_type, double>);
    static_assert(std::is_same_v<Model::activation_type, double>);
    static_assert(std::is_same_v<Model::gradient_type, double>);
    static_assert(std::is_same_v<Model::accumulator_type, double>);
    static_assert(std::is_same_v<Model::optimizer_state_type, double>);
    static_assert(std::is_same_v<Model::loss_type, double>);
    static_assert(Model::parameter_bytes == Model::parameter_count * sizeof(double));
    static_assert(Model::gradient_bytes == Model::gradient_count * sizeof(double));
    static_assert(Model::optimizer_bytes == Model::optimizer_state_count * sizeof(double));
    static_assert(Model::activation_bytes == Model::activation_count * sizeof(double));
    static_assert(Model::workspace_bytes == Model::workspace_count * sizeof(double));

    alignas(Model::alignment)
        std::array<std::byte, Model::required_memory> arena{};
    Model model{edge::external_arena(arena)};
    EDGE_EXPECT_EQ(model.status(), edge::Status::Ok);

    double* p = model.parameter_data();
    p[0] = 2.0;
    p[1] = -1.0;
    p[2] = 0.5;

    const std::array<double, 2> input{3.0, 4.0};
    const std::array<double, 1> upstream{0.25};

    EDGE_EXPECT_EQ(model.forward(input), edge::Status::Ok);
    EDGE_EXPECT_NEAR(model.output()[0], 2.5, 1.0e-12F);

    EDGE_EXPECT_EQ(model.zero_grad(), edge::Status::Ok);
    EDGE_EXPECT_EQ(model.backward(upstream), edge::Status::Ok);

    const double* g = model.gradient_data();
    EDGE_EXPECT_NEAR(g[0], 0.75, 1.0e-12F);
    EDGE_EXPECT_NEAR(g[1], 1.0, 1.0e-12F);
    EDGE_EXPECT_NEAR(g[2], 0.25, 1.0e-12F);

    Model trained;
    EDGE_EXPECT_EQ(trained.status(), edge::Status::Ok);
    double* trained_params = trained.parameter_data();
    trained_params[0] = 0.0;
    trained_params[1] = 0.0;
    trained_params[2] = 0.0;

    edge::Trainer<Model, edge::MSE, edge::SGD> trainer(
        trained,
        edge::TrainerConfig{},
        edge::SGDConfig{.learning_rate = 0.25F});
    const std::array<double, 2> train_input{1.0, 0.0};
    const std::array<double, 1> train_target{1.0};

    EDGE_EXPECT_EQ(trainer.train_step(train_input, train_target), edge::Status::Ok);
    EDGE_EXPECT_NEAR(trainer.last_loss(), 1.0, 1.0e-12F);
    EDGE_EXPECT_NEAR(trained_params[0], 0.5, 1.0e-12F);
    EDGE_EXPECT_NEAR(trained_params[1], 0.0, 1.0e-12F);
    EDGE_EXPECT_NEAR(trained_params[2], 0.5, 1.0e-12F);
}
