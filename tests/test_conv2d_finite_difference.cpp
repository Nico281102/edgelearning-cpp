#include <array>
#include <cstddef>

#include <edge/edge.hpp>

#include "test_harness.hpp"

namespace {

using Model = edge::Model<
    edge::Input<edge::CHW<1, 3, 3>>,
    edge::Conv2D<2, edge::Kernel<2, 2>, edge::Tanh>,
    edge::Flatten,
    edge::Dense<1, edge::Linear>>;

float loss_value(Model& model,
                 const std::array<float, 9>& input,
                 const std::array<float, 1>& target) {
    EDGE_EXPECT_EQ(model.forward(input), edge::Status::Ok);
    return edge::MSE::value(model.output(), edge::TensorView<const float, 1>(target));
}

} // namespace

int main() {
    Model model;
    std::array<float, Model::parameter_count> params{};
    for (std::size_t i = 0; i < params.size(); ++i) {
        params[i] = 0.03F * static_cast<float>((i % 7U) + 1U) - 0.11F;
    }
    EDGE_EXPECT_EQ(model.import_parameters(params), edge::Status::Ok);

    const std::array<float, 9> input{
        0.10F, -0.20F, 0.30F,
        0.40F, 0.05F, -0.15F,
        -0.35F, 0.25F, 0.20F};
    const std::array<float, 1> target{0.12F};

    EDGE_EXPECT_EQ(model.zero_grad(), edge::Status::Ok);
    EDGE_EXPECT_EQ(model.forward(input), edge::Status::Ok);
    std::array<float, 1> output_gradient{};
    edge::TensorView<float, 1> output_gradient_view(output_gradient);
    edge::MSE::evaluate(model.output(), edge::TensorView<const float, 1>(target),
                        output_gradient_view);
    EDGE_EXPECT_EQ(model.backward(output_gradient), edge::Status::Ok);

    const float* analytical = model.gradient_data();
    constexpr float eps = 1.0e-3F;
    for (std::size_t i = 0; i < Model::parameter_count; ++i) {
        std::array<float, Model::parameter_count> plus = params;
        std::array<float, Model::parameter_count> minus = params;
        plus[i] += eps;
        minus[i] -= eps;
        EDGE_EXPECT_EQ(model.import_parameters(plus), edge::Status::Ok);
        const float loss_plus = loss_value(model, input, target);
        EDGE_EXPECT_EQ(model.import_parameters(minus), edge::Status::Ok);
        const float loss_minus = loss_value(model, input, target);
        const float numerical = (loss_plus - loss_minus) / (2.0F * eps);
        EDGE_EXPECT_NEAR(analytical[i], numerical, 3.0e-3F);
    }
}
