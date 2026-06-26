#pragma once

#include <cstddef>
#include <type_traits>

#include <edge/activations.hpp>
#include <edge/backend.hpp>
#include <edge/backends/m55.hpp>
#include <edge/initializers.hpp>
#include <edge/precision.hpp>
#include <edge/tensor.hpp>

namespace edge {

template<std::size_t Features>
struct Input {
    static_assert(Features > 0, "Input feature count must be greater than zero");
    static constexpr std::size_t features = Features;
};

template<
    std::size_t OutFeatures,
    typename Activation = Linear,
    typename Initializer = DefaultInitializer,
    typename PrecisionOverride = UseModelPrecision>
struct Dense {
    static_assert(OutFeatures > 0, "Dense output feature count must be greater than zero");
    static constexpr std::size_t out_features = OutFeatures;
    using activation = Activation;
    using initializer = Initializer;
    using precision_override = PrecisionOverride;
};

namespace detail {

template<std::size_t InFeatures, typename DenseSpec>
struct DenseInstance {
    static constexpr std::size_t in_features = InFeatures;
    static constexpr std::size_t out_features = DenseSpec::out_features;
    using activation = typename DenseSpec::activation;
    using initializer = typename DenseSpec::initializer;
    using precision_override = typename DenseSpec::precision_override;

    static constexpr std::size_t weight_count = in_features * out_features;
    static constexpr std::size_t bias_count = out_features;
    static constexpr std::size_t parameter_count = weight_count + bias_count;
    static constexpr bool stores_preactivation =
        activation::storage == ActivationStorage::PreActivationOnly ||
        activation::storage == ActivationStorage::OutputAndPreActivation;
    static constexpr std::size_t preactivation_count =
        stores_preactivation ? out_features : 0U;
    static constexpr std::size_t cache_count = preactivation_count;
    static constexpr std::size_t workspace_count = 0;

    template<typename Types>
    static void initialize(TensorView<typename Types::ParameterT, parameter_count> params,
                           DeterministicRng& rng,
                           const InitConfig& config) noexcept {
        using ParameterT = typename Types::ParameterT;
        using Initializer = typename DenseSpec::initializer;
        ParameterT* weights = params.data();
        ParameterT* bias = weights + weight_count;
        Initializer::template fill<ParameterT, in_features, out_features>(weights, rng, config);
        for (std::size_t i = 0; i < bias_count; ++i) {
            bias[i] = static_cast<ParameterT>(config.bias);
        }
    }

    template<typename Types>
    static void forward(TensorView<const typename Types::ActivationT, in_features> input,
                        TensorView<typename Types::ActivationT, out_features> output,
                        TensorView<const typename Types::ParameterT, parameter_count> params,
                        TensorView<typename Types::ActivationT, cache_count> cache,
                        TensorView<typename Types::AccumulatorT, workspace_count>) noexcept {
        using ActivationT = typename Types::ActivationT;
        using AccumulatorT = typename Types::AccumulatorT;
        if constexpr (std::is_same_v<typename Types::BackendT, Backend::M55> &&
                      std::is_same_v<typename Types::ParameterT, float> &&
                      std::is_same_v<ActivationT, float> &&
                      std::is_same_v<AccumulatorT, float>) {
            if (m55_dense_forward<in_features, out_features, cache_count, activation>(
                    input, output, params, cache)) {
                return;
            }
        }

        const auto* weights = params.data();
        const auto* bias = params.data() + weight_count;

        for (std::size_t out = 0; out < out_features; ++out) {
            AccumulatorT z = static_cast<AccumulatorT>(bias[out]);
            const std::size_t row = out * in_features;
            for (std::size_t in = 0; in < in_features; ++in) {
                z += static_cast<AccumulatorT>(weights[row + in]) *
                     static_cast<AccumulatorT>(input[in]);
            }
            if constexpr (stores_preactivation) {
                cache[out] = static_cast<ActivationT>(z);
            }
            output[out] = static_cast<ActivationT>(activation::template forward<AccumulatorT>(z));
        }
    }

    template<bool PropagateInputGradient, typename Types>
    static void backward(TensorView<const typename Types::ActivationT, in_features> input,
                         TensorView<const typename Types::ActivationT, out_features> output,
                         TensorView<const typename Types::AccumulatorT, out_features> upstream,
                         TensorView<typename Types::AccumulatorT,
                                    PropagateInputGradient ? in_features : 0U> downstream,
                         TensorView<const typename Types::ParameterT, parameter_count> params,
                         TensorView<typename Types::GradientT, parameter_count> gradients,
                         TensorView<const typename Types::ActivationT, cache_count> cache,
                         TensorView<typename Types::AccumulatorT, workspace_count>) noexcept {
        using AccumulatorT = typename Types::AccumulatorT;
        if constexpr (std::is_same_v<typename Types::BackendT, Backend::M55> &&
                      std::is_same_v<typename Types::ParameterT, float> &&
                      std::is_same_v<typename Types::ActivationT, float> &&
                      std::is_same_v<typename Types::GradientT, float> &&
                      std::is_same_v<AccumulatorT, float>) {
            if constexpr (PropagateInputGradient) {
                if (m55_dense_backward<in_features, out_features, cache_count, activation>(
                        input, output, upstream, downstream, params, gradients, cache)) {
                    return;
                }
            } else {
                if (m55_dense_backward_inputless<in_features,
                                                 out_features,
                                                 cache_count,
                                                 activation>(
                        input, output, upstream, gradients, cache)) {
                    return;
                }
            }
        }

        (void)params;
        auto* grad_weights = gradients.data();
        auto* grad_bias = gradients.data() + weight_count;

        if constexpr (PropagateInputGradient) {
            for (std::size_t i = 0; i < in_features; ++i) {
                downstream[i] = AccumulatorT{0};
            }
        }

        for (std::size_t out = 0; out < out_features; ++out) {
            const AccumulatorT z =
                stores_preactivation ? static_cast<AccumulatorT>(cache[out]) : AccumulatorT{0};
            const AccumulatorT a = static_cast<AccumulatorT>(output[out]);
            const AccumulatorT deriv = activation_derivative<activation, AccumulatorT>(z, a);
            const AccumulatorT delta = upstream[out] * deriv;
            if (delta == AccumulatorT{0}) {
                continue;
            }
            grad_bias[out] = static_cast<typename Types::GradientT>(
                static_cast<AccumulatorT>(grad_bias[out]) + delta);
            const std::size_t row = out * in_features;
            for (std::size_t in = 0; in < in_features; ++in) {
                grad_weights[row + in] = static_cast<typename Types::GradientT>(
                    static_cast<AccumulatorT>(grad_weights[row + in]) +
                    delta * static_cast<AccumulatorT>(input[in]));
                if constexpr (PropagateInputGradient) {
                    downstream[in] += static_cast<AccumulatorT>(params[row + in]) * delta;
                }
            }
        }
    }
};

} // namespace detail

} // namespace edge
