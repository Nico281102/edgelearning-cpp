#include <array>
#include <cstddef>

#include <edge/edge.hpp>

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

int main() {
    using Model = edge::Model<
        edge::Input<2>,
        TrainableScale,
        edge::Dense<1>>;

    Model model;
    model.initialize(edge::InitConfig{.seed = 5U});

    const std::array<float, 2> input{0.5F, -1.0F};
    return edge::is_ok(model.forward(input)) ? 0 : 1;
}
