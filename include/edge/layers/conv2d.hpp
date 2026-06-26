#pragma once

#include <cstddef>
#include <type_traits>

#include <edge/activations.hpp>
#include <edge/backend.hpp>
#include <edge/initializers.hpp>
#include <edge/tensor.hpp>
#include <edge/tensor_spec.hpp>

namespace edge {

template<std::size_t Height, std::size_t Width = Height>
struct Kernel {
    static_assert(Height > 0, "Kernel height must be greater than zero");
    static_assert(Width > 0, "Kernel width must be greater than zero");

    static constexpr std::size_t height = Height;
    static constexpr std::size_t width = Width;
};

template<std::size_t Height, std::size_t Width = Height>
struct Stride {
    static_assert(Height > 0, "Stride height must be greater than zero");
    static_assert(Width > 0, "Stride width must be greater than zero");

    static constexpr std::size_t height = Height;
    static constexpr std::size_t width = Width;
};

template<std::size_t Height, std::size_t Width = Height>
struct Padding {
    static constexpr std::size_t height = Height;
    static constexpr std::size_t width = Width;
};

template<
    std::size_t OutChannels,
    typename KernelSpec,
    typename Activation = Linear,
    typename Initializer = DefaultInitializer,
    typename StrideSpec = Stride<1>,
    typename PaddingSpec = Padding<0>>
struct Conv2D {
    static_assert(OutChannels > 0, "Conv2D output channel count must be greater than zero");

    static constexpr std::size_t out_channels = OutChannels;
    static constexpr std::size_t kernel_height = KernelSpec::height;
    static constexpr std::size_t kernel_width = KernelSpec::width;
    static constexpr std::size_t stride_height = StrideSpec::height;
    static constexpr std::size_t stride_width = StrideSpec::width;
    static constexpr std::size_t padding_height = PaddingSpec::height;
    static constexpr std::size_t padding_width = PaddingSpec::width;
    using activation = Activation;
    using initializer = Initializer;

    template<typename InputSpec>
    struct Instance {
        static_assert(InputSpec::layout == Layout::CHW,
                      "Conv2D requires a CHW input spec");
        static_assert(InputSpec::rank == 3U,
                      "Conv2D CHW input spec must have rank 3");

        using input_spec = InputSpec;

        static constexpr std::size_t in_channels =
            shape_dim_v<0U, typename input_spec::shape>;
        static constexpr std::size_t in_height =
            shape_dim_v<1U, typename input_spec::shape>;
        static constexpr std::size_t in_width =
            shape_dim_v<2U, typename input_spec::shape>;
        static_assert(in_height + 2U * padding_height >= kernel_height,
                      "Conv2D kernel height exceeds padded input height");
        static_assert(in_width + 2U * padding_width >= kernel_width,
                      "Conv2D kernel width exceeds padded input width");

        static constexpr std::size_t out_height =
            ((in_height + 2U * padding_height - kernel_height) / stride_height) + 1U;
        static constexpr std::size_t out_width =
            ((in_width + 2U * padding_width - kernel_width) / stride_width) + 1U;
        using output_spec = CHW<OutChannels, out_height, out_width>;

        static constexpr std::size_t in_features = input_spec::elements;
        static constexpr std::size_t out_features = output_spec::elements;
        static constexpr std::size_t filter_count =
            OutChannels * in_channels * kernel_height * kernel_width;
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
                                       in_channels * kernel_height * kernel_width,
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
            if constexpr (!std::is_same_v<typename Types::BackendT, Backend::Generic> &&
                          !backend_falls_back_to_generic_v<typename Types::BackendT>) {
                static_assert(detail::always_false_v<typename Types::BackendT>,
                              "Selected backend does not provide Conv2D forward and generic "
                              "fallback is disabled");
            }

            using ActivationT = typename Types::ActivationT;
            using AccumulatorT = typename Types::AccumulatorT;
            const auto* filters = params.data();
            const auto* bias = params.data() + filter_count;

            for (std::size_t oc = 0; oc < OutChannels; ++oc) {
                for (std::size_t oh = 0; oh < out_height; ++oh) {
                    for (std::size_t ow = 0; ow < out_width; ++ow) {
                        AccumulatorT z = static_cast<AccumulatorT>(bias[oc]);
                        for (std::size_t ic = 0; ic < in_channels; ++ic) {
                            for (std::size_t kh = 0; kh < kernel_height; ++kh) {
                                for (std::size_t kw = 0; kw < kernel_width; ++kw) {
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
            if constexpr (!std::is_same_v<typename Types::BackendT, Backend::Generic> &&
                          !backend_falls_back_to_generic_v<typename Types::BackendT>) {
                static_assert(detail::always_false_v<typename Types::BackendT>,
                              "Selected backend does not provide Conv2D backward and generic "
                              "fallback is disabled");
            }

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

                        for (std::size_t ic = 0; ic < in_channels; ++ic) {
                            for (std::size_t kh = 0; kh < kernel_height; ++kh) {
                                for (std::size_t kw = 0; kw < kernel_width; ++kw) {
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
            return static_cast<std::ptrdiff_t>(out_row * stride_height + kernel_row) -
                   static_cast<std::ptrdiff_t>(padding_height);
        }

        static constexpr std::ptrdiff_t input_col(std::size_t out_col,
                                                  std::size_t kernel_col) noexcept {
            return static_cast<std::ptrdiff_t>(out_col * stride_width + kernel_col) -
                   static_cast<std::ptrdiff_t>(padding_width);
        }

        static constexpr bool is_valid_input(std::ptrdiff_t row,
                                             std::ptrdiff_t col) noexcept {
            return row >= 0 && col >= 0 &&
                   row < static_cast<std::ptrdiff_t>(in_height) &&
                   col < static_cast<std::ptrdiff_t>(in_width);
        }

        static constexpr std::size_t input_index(std::size_t channel,
                                                 std::size_t row,
                                                 std::size_t col) noexcept {
            return (channel * in_height + row) * in_width + col;
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
            return ((out_channel * in_channels + in_channel) * kernel_height + kernel_row) *
                       kernel_width +
                   kernel_col;
        }
    };
};

} // namespace edge
