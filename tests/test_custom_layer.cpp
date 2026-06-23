#include <array>
#include <cstddef>

#include <edge/edge.hpp>

#include "test_harness.hpp"

namespace {

struct TrainableScale {
    template<std::size_t InFeatures>
    struct Instance {
        static constexpr std::size_t in_features = InFeatures;
        static constexpr std::size_t out_features = InFeatures;
        static constexpr std::size_t parameter_count = InFeatures;
        static constexpr std::size_t cache_count = 0;
        static constexpr std::size_t workspace_count = 0;

        template<typename Types>
        static void initialize(edge::TensorView<typename Types::ParameterT, parameter_count> params,
                               edge::DeterministicRng&,
                               const edge::InitConfig&) noexcept {
            for (std::size_t i = 0; i < parameter_count; ++i) {
                params[i] = typename Types::ParameterT{1};
            }
        }

        template<typename Types>
        static void forward(
            edge::TensorView<const typename Types::ActivationT, in_features> input,
            edge::TensorView<typename Types::ActivationT, out_features> output,
            edge::TensorView<const typename Types::ParameterT, parameter_count> params,
            edge::TensorView<typename Types::ActivationT, cache_count>,
            edge::TensorView<typename Types::AccumulatorT, workspace_count>) noexcept {
            for (std::size_t i = 0; i < in_features; ++i) {
                output[i] = static_cast<typename Types::ActivationT>(
                    static_cast<typename Types::AccumulatorT>(input[i]) *
                    static_cast<typename Types::AccumulatorT>(params[i]));
            }
        }

        template<typename Types>
        static void backward(
            edge::TensorView<const typename Types::ActivationT, in_features> input,
            edge::TensorView<const typename Types::ActivationT, out_features>,
            edge::TensorView<const typename Types::AccumulatorT, out_features> upstream,
            edge::TensorView<typename Types::AccumulatorT, in_features> downstream,
            edge::TensorView<const typename Types::ParameterT, parameter_count> params,
            edge::TensorView<typename Types::GradientT, parameter_count> gradients,
            edge::TensorView<const typename Types::ActivationT, cache_count>,
            edge::TensorView<typename Types::AccumulatorT, workspace_count>) noexcept {
            for (std::size_t i = 0; i < in_features; ++i) {
                gradients[i] = static_cast<typename Types::GradientT>(
                    static_cast<typename Types::AccumulatorT>(gradients[i]) +
                    upstream[i] * static_cast<typename Types::AccumulatorT>(input[i]));
                downstream[i] =
                    upstream[i] * static_cast<typename Types::AccumulatorT>(params[i]);
            }
        }
    };
};

} // namespace

int main() {
    using Model = edge::Model<
        edge::Input<2>,
        TrainableScale,
        edge::Dense<1, edge::Linear>>;

    static_assert(Model::input_size == 2);
    static_assert(Model::output_size == 1);
    static_assert(Model::layer_count == 2);
    static_assert(Model::parameter_count == 5);

    Model model;
    EDGE_EXPECT_EQ(model.status(), edge::Status::Ok);

    float* p = model.parameter_data();
    p[0] = 2.0F;
    p[1] = 3.0F;
    p[2] = 1.0F;
    p[3] = 1.0F;
    p[4] = 0.0F;

    const std::array<float, 2> input{4.0F, 5.0F};
    const std::array<float, 1> upstream{1.0F};

    EDGE_EXPECT_EQ(model.forward(input), edge::Status::Ok);
    EDGE_EXPECT_NEAR(model.output()[0], 23.0F, 1.0e-6F);

    EDGE_EXPECT_EQ(model.zero_grad(), edge::Status::Ok);
    EDGE_EXPECT_EQ(model.backward(upstream), edge::Status::Ok);

    const float* g = model.gradient_data();
    EDGE_EXPECT_NEAR(g[0], 4.0F, 1.0e-6F);
    EDGE_EXPECT_NEAR(g[1], 5.0F, 1.0e-6F);
    EDGE_EXPECT_NEAR(g[2], 8.0F, 1.0e-6F);
    EDGE_EXPECT_NEAR(g[3], 15.0F, 1.0e-6F);
    EDGE_EXPECT_NEAR(g[4], 1.0F, 1.0e-6F);
}
