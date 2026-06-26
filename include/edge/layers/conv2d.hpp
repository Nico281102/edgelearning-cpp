#pragma once

#include <cstddef>

#include <edge/activations.hpp>
#include <edge/initializers.hpp>
#include <edge/tensor.hpp>

namespace edge {

template<
    std::size_t InChannels,
    std::size_t InHeight,
    std::size_t InWidth,
    std::size_t OutChannels,
    std::size_t KernelHeight,
    std::size_t KernelWidth,
    typename Activation = Linear,
    typename Initializer = DefaultInitializer,
    std::size_t StrideHeight = 1,
    std::size_t StrideWidth = StrideHeight,
    std::size_t PaddingHeight = 0,
    std::size_t PaddingWidth = PaddingHeight>
struct Conv2D {
    static_assert(InChannels > 0, "Conv2D input channel count must be greater than zero");
    static_assert(InHeight > 0, "Conv2D input height must be greater than zero");
    static_assert(InWidth > 0, "Conv2D input width must be greater than zero");
    static_assert(OutChannels > 0, "Conv2D output channel count must be greater than zero");
    static_assert(KernelHeight > 0, "Conv2D kernel height must be greater than zero");
    static_assert(KernelWidth > 0, "Conv2D kernel width must be greater than zero");
    static_assert(StrideHeight > 0, "Conv2D stride height must be greater than zero");
    static_assert(StrideWidth > 0, "Conv2D stride width must be greater than zero");
    static_assert(InHeight + 2U * PaddingHeight >= KernelHeight,
                  "Conv2D kernel height exceeds padded input height");
    static_assert(InWidth + 2U * PaddingWidth >= KernelWidth,
                  "Conv2D kernel width exceeds padded input width");

    static constexpr std::size_t in_channels = InChannels;
    static constexpr std::size_t in_height = InHeight;
    static constexpr std::size_t in_width = InWidth;
    static constexpr std::size_t out_channels = OutChannels;
    static constexpr std::size_t kernel_height = KernelHeight;
    static constexpr std::size_t kernel_width = KernelWidth;
    static constexpr std::size_t stride_height = StrideHeight;
    static constexpr std::size_t stride_width = StrideWidth;
    static constexpr std::size_t padding_height = PaddingHeight;
    static constexpr std::size_t padding_width = PaddingWidth;
    static constexpr std::size_t out_height =
        ((InHeight + 2U * PaddingHeight - KernelHeight) / StrideHeight) + 1U;
    static constexpr std::size_t out_width =
        ((InWidth + 2U * PaddingWidth - KernelWidth) / StrideWidth) + 1U;
    using activation = Activation;
    using initializer = Initializer;

    template<std::size_t InFeatures>
    struct Instance {
        static_assert(InFeatures == InChannels * InHeight * InWidth,
                      "Conv2D input feature count must match InChannels * InHeight * InWidth");

        static constexpr std::size_t in_features = InFeatures;
        static constexpr std::size_t out_features = OutChannels * out_height * out_width;
        static constexpr std::size_t filter_count =
            OutChannels * InChannels * KernelHeight * KernelWidth;
        static constexpr std::size_t bias_count = OutChannels;
        static constexpr std::size_t parameter_count = filter_count + bias_count;
        static constexpr bool stores_preactivation =
            Activation::storage == ActivationStorage::PreActivationOnly ||
            Activation::storage == ActivationStorage::OutputAndPreActivation;
        static constexpr std::size_t cache_count =
            stores_preactivation ? out_features : 0U;
        static constexpr std::size_t workspace_count = 0;

        template<typename Types>
        static void initialize(TensorView<typename Types::ParameterT, parameter_count> params,
                               DeterministicRng& rng,
                               const InitConfig& config) noexcept {
            using ParameterT = typename Types::ParameterT;
            ParameterT* filters = params.data();
            ParameterT* bias = filters + filter_count;
            Initializer::template fill<ParameterT,
                                       InChannels * KernelHeight * KernelWidth,
                                       OutChannels>(filters, rng, config);
            for (std::size_t i = 0; i < bias_count; ++i) {
                bias[i] = static_cast<ParameterT>(config.bias);
            }
        }

        template<typename Types>
        static void forward(
            TensorView<const typename Types::ActivationT, in_features> input,
            TensorView<typename Types::ActivationT, out_features> output,
            TensorView<const typename Types::ParameterT, parameter_count> params,
            TensorView<typename Types::ActivationT, cache_count> cache,
            TensorView<typename Types::AccumulatorT, workspace_count>) noexcept {
            using ActivationT = typename Types::ActivationT;
            using AccumulatorT = typename Types::AccumulatorT;
            const auto* filters = params.data();
            const auto* bias = params.data() + filter_count;

            for (std::size_t oc = 0; oc < OutChannels; ++oc) {
                for (std::size_t oh = 0; oh < out_height; ++oh) {
                    for (std::size_t ow = 0; ow < out_width; ++ow) {
                        AccumulatorT z = static_cast<AccumulatorT>(bias[oc]);
                        for (std::size_t ic = 0; ic < InChannels; ++ic) {
                            for (std::size_t kh = 0; kh < KernelHeight; ++kh) {
                                for (std::size_t kw = 0; kw < KernelWidth; ++kw) {
                                    const auto ih = input_row(oh, kh);
                                    const auto iw = input_col(ow, kw);
                                    if (!is_valid_input(ih, iw)) {
                                        continue;
                                    }
                                    z += static_cast<AccumulatorT>(
                                             filters[filter_index(oc, ic, kh, kw)]) *
                                         static_cast<AccumulatorT>(
                                             input[input_index(ic,
                                                               static_cast<std::size_t>(ih),
                                                               static_cast<std::size_t>(iw))]);
                                }
                            }
                        }
                        const std::size_t out_idx = output_index(oc, oh, ow);
                        if constexpr (stores_preactivation) {
                            cache[out_idx] = static_cast<ActivationT>(z);
                        }
                        output[out_idx] =
                            static_cast<ActivationT>(Activation::template forward<AccumulatorT>(z));
                    }
                }
            }
        }

        template<bool PropagateInputGradient, typename Types>
        static void backward(
            TensorView<const typename Types::ActivationT, in_features> input,
            TensorView<const typename Types::ActivationT, out_features> output,
            TensorView<const typename Types::AccumulatorT, out_features> upstream,
            TensorView<typename Types::AccumulatorT,
                       PropagateInputGradient ? in_features : 0U> downstream,
            TensorView<const typename Types::ParameterT, parameter_count> params,
            TensorView<typename Types::GradientT, parameter_count> gradients,
            TensorView<const typename Types::ActivationT, cache_count> cache,
            TensorView<typename Types::AccumulatorT, workspace_count>) noexcept {
            using AccumulatorT = typename Types::AccumulatorT;
            (void)params;
            auto* grad_filters = gradients.data();
            auto* grad_bias = gradients.data() + filter_count;

            if constexpr (PropagateInputGradient) {
                for (std::size_t i = 0; i < in_features; ++i) {
                    downstream[i] = AccumulatorT{0};
                }
            }

            for (std::size_t oc = 0; oc < OutChannels; ++oc) {
                for (std::size_t oh = 0; oh < out_height; ++oh) {
                    for (std::size_t ow = 0; ow < out_width; ++ow) {
                        const std::size_t out_idx = output_index(oc, oh, ow);
                        const AccumulatorT z =
                            stores_preactivation ? static_cast<AccumulatorT>(cache[out_idx])
                                                 : AccumulatorT{0};
                        const AccumulatorT a = static_cast<AccumulatorT>(output[out_idx]);
                        const AccumulatorT deriv =
                            activation_derivative<Activation, AccumulatorT>(z, a);
                        const AccumulatorT delta = upstream[out_idx] * deriv;
                        grad_bias[oc] = static_cast<typename Types::GradientT>(
                            static_cast<AccumulatorT>(grad_bias[oc]) + delta);

                        for (std::size_t ic = 0; ic < InChannels; ++ic) {
                            for (std::size_t kh = 0; kh < KernelHeight; ++kh) {
                                for (std::size_t kw = 0; kw < KernelWidth; ++kw) {
                                    const auto ih = input_row(oh, kh);
                                    const auto iw = input_col(ow, kw);
                                    if (!is_valid_input(ih, iw)) {
                                        continue;
                                    }
                                    const std::size_t in_idx =
                                        input_index(ic,
                                                    static_cast<std::size_t>(ih),
                                                    static_cast<std::size_t>(iw));
                                    const std::size_t f_idx = filter_index(oc, ic, kh, kw);
                                    grad_filters[f_idx] = static_cast<typename Types::GradientT>(
                                        static_cast<AccumulatorT>(grad_filters[f_idx]) +
                                        delta * static_cast<AccumulatorT>(input[in_idx]));
                                    if constexpr (PropagateInputGradient) {
                                        downstream[in_idx] +=
                                            static_cast<AccumulatorT>(params[f_idx]) * delta;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    private:
        static constexpr std::ptrdiff_t input_row(std::size_t out_row,
                                                  std::size_t kernel_row) noexcept {
            return static_cast<std::ptrdiff_t>(out_row * StrideHeight + kernel_row) -
                   static_cast<std::ptrdiff_t>(PaddingHeight);
        }

        static constexpr std::ptrdiff_t input_col(std::size_t out_col,
                                                  std::size_t kernel_col) noexcept {
            return static_cast<std::ptrdiff_t>(out_col * StrideWidth + kernel_col) -
                   static_cast<std::ptrdiff_t>(PaddingWidth);
        }

        static constexpr bool is_valid_input(std::ptrdiff_t row,
                                             std::ptrdiff_t col) noexcept {
            return row >= 0 && col >= 0 &&
                   row < static_cast<std::ptrdiff_t>(InHeight) &&
                   col < static_cast<std::ptrdiff_t>(InWidth);
        }

        static constexpr std::size_t input_index(std::size_t channel,
                                                 std::size_t row,
                                                 std::size_t col) noexcept {
            return (channel * InHeight + row) * InWidth + col;
        }

        static constexpr std::size_t output_index(std::size_t channel,
                                                  std::size_t row,
                                                  std::size_t col) noexcept {
            return (channel * out_height + row) * out_width + col;
        }

        static constexpr std::size_t filter_index(std::size_t out_channel,
                                                  std::size_t in_channel,
                                                  std::size_t kernel_row,
                                                  std::size_t kernel_col) noexcept {
            return ((out_channel * InChannels + in_channel) * KernelHeight + kernel_row) *
                       KernelWidth +
                   kernel_col;
        }
    };
};

} // namespace edge
