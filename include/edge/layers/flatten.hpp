#pragma once

#include <cstddef>

#include <edge/initializers.hpp>
#include <edge/tensor.hpp>
#include <edge/tensor_spec.hpp>

namespace edge {

struct Flatten {
    template<typename InputSpec>
    struct Instance {
        using input_spec = InputSpec;
        using output_spec = Vector<InputSpec::elements>;

        static constexpr std::size_t in_features = input_spec::elements;
        static constexpr std::size_t out_features = output_spec::elements;
        static constexpr std::size_t parameter_count = 0;
        static constexpr std::size_t cache_count = 0;
        static constexpr std::size_t workspace_count = 0;

        template<typename Types>
        static void initialize(TensorView<typename Types::ParameterT, parameter_count>,
                               DeterministicRng&,
                               const InitConfig&) noexcept {}

        template<typename Types>
        static void forward(TensorView<const typename Types::ActivationT, in_features> input,
                            TensorView<typename Types::ActivationT, out_features> output,
                            TensorView<const typename Types::ParameterT, parameter_count>,
                            TensorView<typename Types::ActivationT, cache_count>,
                            TensorView<typename Types::AccumulatorT, workspace_count>) noexcept {
            for (std::size_t i = 0; i < in_features; ++i) {
                output[i] = input[i];
            }
        }

        template<bool PropagateInputGradient, typename Types>
        static void backward(
            TensorView<const typename Types::ActivationT, in_features>,
            TensorView<const typename Types::ActivationT, out_features>,
            TensorView<const typename Types::AccumulatorT, out_features> upstream,
            TensorView<typename Types::AccumulatorT,
                       PropagateInputGradient ? in_features : 0U> downstream,
            TensorView<const typename Types::ParameterT, parameter_count>,
            TensorView<typename Types::GradientT, parameter_count>,
            TensorView<const typename Types::ActivationT, cache_count>,
            TensorView<typename Types::AccumulatorT, workspace_count>) noexcept {
            if constexpr (PropagateInputGradient) {
                for (std::size_t i = 0; i < in_features; ++i) {
                    downstream[i] = upstream[i];
                }
            }
        }
    };
};

} // namespace edge
