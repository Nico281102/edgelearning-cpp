#include "el_cvscpp_ablation_run.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "main.h"

#include <edge/edge.hpp>

extern "C" {
#include "edgelearning.h"
}

#ifndef EL_CVSCPP_H1
#error "EL_CVSCPP_H1 must be defined"
#endif

#ifndef EL_CVSCPP_H2
#error "EL_CVSCPP_H2 must be defined"
#endif

#ifndef EL_CVSCPP_INPUT_FEATURES
#define EL_CVSCPP_INPUT_FEATURES 3
#endif

#ifndef EL_CVSCPP_VARIANT
#define EL_CVSCPP_VARIANT 0
#endif

#define EL_CVSCPP_VARIANT_ALL 0
#define EL_CVSCPP_VARIANT_LEGACY_C 1
#define EL_CVSCPP_VARIANT_CPP_DIRECT_C_BACKEND 2
#define EL_CVSCPP_VARIANT_CPP_M55 3
#define EL_CVSCPP_VARIANT_CPP_GENERIC 4
#define EL_CVSCPP_VARIANT_RLTOOLS_GENERIC 5

#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_RLTOOLS_GENERIC
#define EL_CVSCPP_ENABLE_RLTOOLS 1
#else
#define EL_CVSCPP_ENABLE_RLTOOLS 0
#endif

#if EL_CVSCPP_ENABLE_RLTOOLS
/*
 * STM32N6 CMSIS headers define RNG as a peripheral macro. RLTools uses RNG as a
 * template parameter name, so the macro must be removed before RLTools headers.
 */
#ifdef RNG
#undef RNG
#endif
#ifndef RL_TOOLS_DEVICES_DISABLE_REDEFINITION_DETECTION
#define RL_TOOLS_DEVICES_DISABLE_REDEFINITION_DETECTION
#endif
#include <rl_tools/rl_tools.h>
#include <rl_tools/operations/arm/group_1.h>
#include <rl_tools/operations/arm/group_2.h>
#include <rl_tools/operations/arm/group_3.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/loss_functions/mse/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#endif

extern UART_HandleTypeDef hlpuart1;

namespace {

constexpr std::size_t kInputFeatures = EL_CVSCPP_INPUT_FEATURES;
constexpr std::size_t kHidden1 = EL_CVSCPP_H1;
constexpr std::size_t kHidden2 = EL_CVSCPP_H2;
constexpr std::size_t kOutputFeatures = 1;
static_assert(kInputFeatures > 0U);
constexpr std::size_t kBatchSize = 256;
constexpr std::size_t kRolloutSamples = 1024;
constexpr std::size_t kEpochs = 2;
static_assert((kRolloutSamples % kBatchSize) == 0U);
constexpr std::size_t kBatchesPerEpoch = kRolloutSamples / kBatchSize;
constexpr std::size_t kOptimizerSteps = kEpochs * kBatchesPerEpoch;
constexpr std::size_t kSamplePasses = kEpochs * kRolloutSamples;
constexpr std::size_t kSeedCount = 10;
constexpr std::size_t kWarmupRuns = 2;
constexpr std::uint32_t kConvergenceTraceSeed = 0;
constexpr float kLearningRate = 1.0e-3F;
constexpr float kAdamBeta1 = 0.9F;
constexpr float kAdamBeta2 = 0.999F;
constexpr float kAdamEpsilon = 1.0e-8F;
constexpr float kTolerance = 1.0e-3F;
constexpr edge::AdamConfig kAdamConfig{
    .learning_rate = kLearningRate,
    .beta1 = kAdamBeta1,
    .beta2 = kAdamBeta2,
    .epsilon = kAdamEpsilon,
};

#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL
constexpr const char* kVariantName = "all";
#elif EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_LEGACY_C
constexpr const char* kVariantName = "legacy_c";
#elif EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_DIRECT_C_BACKEND
constexpr const char* kVariantName = "cpp_direct_c_backend";
#elif EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_M55
constexpr const char* kVariantName = "cpp_m55";
#elif EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_GENERIC
constexpr const char* kVariantName = "cpp_generic";
#elif EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_RLTOOLS_GENERIC
constexpr const char* kVariantName = "rltools_generic";
#else
#error "Unsupported EL_CVSCPP_VARIANT"
#endif

#define EL_CVSCPP_STR_HELPER(value) #value
#define EL_CVSCPP_STR(value) EL_CVSCPP_STR_HELPER(value)

using GenericModel = edge::Model<
    edge::Backend::Generic,
    edge::Input<kInputFeatures>,
    edge::Dense<kHidden1, edge::ReLU>,
    edge::Dense<kHidden2, edge::ReLU>,
    edge::Dense<kOutputFeatures, edge::Linear>>;

using M55Model = edge::Model<
    edge::Backend::M55,
    edge::Input<kInputFeatures>,
    edge::Dense<kHidden1, edge::ReLU>,
    edge::Dense<kHidden2, edge::ReLU>,
    edge::Dense<kOutputFeatures, edge::Linear>>;

struct LegacyCLinear {};
struct LegacyCReLU {};

template<std::size_t OutFeatures, typename Activation>
struct LegacyCDense {
    template<std::size_t InFeatures>
    struct Instance {
        static constexpr std::size_t in_features = InFeatures;
        static constexpr std::size_t out_features = OutFeatures;
        static constexpr std::size_t weight_count = in_features * out_features;
        static constexpr std::size_t bias_count = out_features;
        static constexpr std::size_t parameter_count = weight_count + bias_count;
        static constexpr std::size_t cache_count = 0;
        static constexpr std::size_t workspace_count = 0;

        template<typename Types>
        static void initialize(edge::TensorView<typename Types::ParameterT, parameter_count> params,
                               edge::DeterministicRng&,
                               const edge::InitConfig&) noexcept {
            for (std::size_t i = 0; i < parameter_count; ++i) {
                params[i] = typename Types::ParameterT{0};
            }
        }

        template<typename Types>
        static void forward(
            edge::TensorView<const typename Types::ActivationT, in_features> input,
            edge::TensorView<typename Types::ActivationT, out_features> output,
            edge::TensorView<const typename Types::ParameterT, parameter_count> params,
            edge::TensorView<typename Types::ActivationT, cache_count>,
            edge::TensorView<typename Types::AccumulatorT, workspace_count>) noexcept {
            static_assert(std::is_same_v<typename Types::ParameterT, float>);
            static_assert(std::is_same_v<typename Types::ActivationT, float>);

            el_tensor_t input_t{
                .data = const_cast<float*>(input.data()),
                .rows = static_cast<std::uint16_t>(in_features),
                .cols = 1U,
            };
            el_tensor_t weights_t{
                .data = const_cast<float*>(params.data()),
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = static_cast<std::uint16_t>(in_features),
            };
            el_tensor_t bias_t{
                .data = const_cast<float*>(params.data() + weight_count),
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = 1U,
            };
            el_tensor_t output_t{
                .data = output.data(),
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = 1U,
            };
            el_layer_dense_t layer{};
            layer.weights = &weights_t;
            layer.bias = &bias_t;
            layer.output = &output_t;
            layer.activation =
                std::is_same_v<Activation, LegacyCReLU> ? EL_ACT_RELU : EL_ACT_NONE;

            if constexpr (std::is_same_v<Activation, LegacyCReLU>) {
                el_backend_dense_forward_relu(&input_t, &layer, &output_t);
            } else {
                alignas(EL_ALIGNMENT_BYTES) std::array<float, out_features> aligned_output{};
                el_tensor_t aligned_output_t{
                    .data = aligned_output.data(),
                    .rows = static_cast<std::uint16_t>(out_features),
                    .cols = 1U,
                };
                layer.output = &aligned_output_t;
                el_backend_dense_forward_linear(&input_t, &layer, &aligned_output_t);
                for (std::size_t i = 0; i < out_features; ++i) {
                    output[i] = aligned_output[i];
                }
            }
        }

        template<typename Types>
        static void backward(
            edge::TensorView<const typename Types::ActivationT, in_features> input,
            edge::TensorView<const typename Types::ActivationT, out_features> output,
            edge::TensorView<const typename Types::AccumulatorT, out_features> upstream,
            edge::TensorView<typename Types::AccumulatorT, in_features> downstream,
            edge::TensorView<const typename Types::ParameterT, parameter_count> params,
            edge::TensorView<typename Types::GradientT, parameter_count> gradients,
            edge::TensorView<const typename Types::ActivationT, cache_count>,
            edge::TensorView<typename Types::AccumulatorT, workspace_count>) noexcept {
            static_assert(std::is_same_v<typename Types::ParameterT, float>);
            static_assert(std::is_same_v<typename Types::ActivationT, float>);
            static_assert(std::is_same_v<typename Types::AccumulatorT, float>);
            static_assert(std::is_same_v<typename Types::GradientT, float>);

            el_tensor_t input_t{
                .data = const_cast<float*>(input.data()),
                .rows = static_cast<std::uint16_t>(in_features),
                .cols = 1U,
            };
            el_tensor_t output_t{
                .data = const_cast<float*>(output.data()),
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = 1U,
            };
            el_tensor_t upstream_t{
                .data = const_cast<float*>(upstream.data()),
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = 1U,
            };
            el_tensor_t downstream_t{
                .data = downstream.data(),
                .rows = static_cast<std::uint16_t>(in_features),
                .cols = 1U,
            };
            el_tensor_t weights_t{
                .data = const_cast<float*>(params.data()),
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = static_cast<std::uint16_t>(in_features),
            };
            el_tensor_t bias_t{
                .data = const_cast<float*>(params.data() + weight_count),
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = 1U,
            };
            el_tensor_t grad_weights_t{
                .data = gradients.data(),
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = static_cast<std::uint16_t>(in_features),
            };
            el_tensor_t grad_bias_t{
                .data = gradients.data() + weight_count,
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = 1U,
            };

            el_layer_dense_t layer{};
            layer.weights = &weights_t;
            layer.bias = &bias_t;
            layer.output = &output_t;
            layer.activation =
                std::is_same_v<Activation, LegacyCReLU> ? EL_ACT_RELU : EL_ACT_NONE;
            layer.grad_weights = &grad_weights_t;
            layer.grad_bias = &grad_bias_t;

            if constexpr (std::is_same_v<Activation, LegacyCReLU>) {
                el_backend_dense_backward_relu(
                    &layer, &input_t, &upstream_t, &downstream_t, &grad_weights_t, &grad_bias_t);
            } else {
                el_backend_dense_backward_linear(
                    &layer, &input_t, &upstream_t, &downstream_t, &grad_weights_t, &grad_bias_t);
            }
        }

        template<typename Types>
        static void backward_inputless(
            edge::TensorView<const typename Types::ActivationT, in_features> input,
            edge::TensorView<const typename Types::ActivationT, out_features> output,
            edge::TensorView<const typename Types::AccumulatorT, out_features> upstream,
            edge::TensorView<const typename Types::ParameterT, parameter_count> params,
            edge::TensorView<typename Types::GradientT, parameter_count> gradients,
            edge::TensorView<const typename Types::ActivationT, cache_count>,
            edge::TensorView<typename Types::AccumulatorT, workspace_count>) noexcept {
            static_assert(std::is_same_v<typename Types::ParameterT, float>);
            static_assert(std::is_same_v<typename Types::ActivationT, float>);
            static_assert(std::is_same_v<typename Types::AccumulatorT, float>);
            static_assert(std::is_same_v<typename Types::GradientT, float>);

            el_tensor_t input_t{
                .data = const_cast<float*>(input.data()),
                .rows = static_cast<std::uint16_t>(in_features),
                .cols = 1U,
            };
            el_tensor_t output_t{
                .data = const_cast<float*>(output.data()),
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = 1U,
            };
            el_tensor_t upstream_t{
                .data = const_cast<float*>(upstream.data()),
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = 1U,
            };
            el_tensor_t weights_t{
                .data = const_cast<float*>(params.data()),
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = static_cast<std::uint16_t>(in_features),
            };
            el_tensor_t bias_t{
                .data = const_cast<float*>(params.data() + weight_count),
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = 1U,
            };
            el_tensor_t grad_weights_t{
                .data = gradients.data(),
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = static_cast<std::uint16_t>(in_features),
            };
            el_tensor_t grad_bias_t{
                .data = gradients.data() + weight_count,
                .rows = static_cast<std::uint16_t>(out_features),
                .cols = 1U,
            };

            el_layer_dense_t layer{};
            layer.weights = &weights_t;
            layer.bias = &bias_t;
            layer.output = &output_t;
            layer.activation =
                std::is_same_v<Activation, LegacyCReLU> ? EL_ACT_RELU : EL_ACT_NONE;
            layer.grad_weights = &grad_weights_t;
            layer.grad_bias = &grad_bias_t;

            if constexpr (std::is_same_v<Activation, LegacyCReLU>) {
                el_backend_dense_backward_relu(
                    &layer, &input_t, &upstream_t, nullptr, &grad_weights_t, &grad_bias_t);
            } else {
                el_backend_dense_backward_linear(
                    &layer, &input_t, &upstream_t, nullptr, &grad_weights_t, &grad_bias_t);
            }
        }
    };
};

using DirectCBackendModel = edge::Model<
    edge::Backend::M55,
    edge::Input<kInputFeatures>,
    LegacyCDense<kHidden1, LegacyCReLU>,
    LegacyCDense<kHidden2, LegacyCReLU>,
    LegacyCDense<kOutputFeatures, LegacyCLinear>>;

static_assert(M55Model::parameter_count == DirectCBackendModel::parameter_count);
static_assert(M55Model::parameter_count == GenericModel::parameter_count);
static_assert(M55Model::input_size == kInputFeatures);
static_assert(M55Model::output_size == kOutputFeatures);

#if EL_CVSCPP_ENABLE_RLTOOLS
namespace rlt = rl_tools;

using RltoolsDevice = rlt::devices::arm::OPT<rlt::devices::DefaultARMSpecification>;
using RltoolsTypePolicy = rlt::numeric_types::Policy<float>;
using RltoolsTI = typename RltoolsDevice::index_t;
using RltoolsRng = typename RltoolsDevice::SPEC::RANDOM::template ENGINE<>;

struct RltoolsAdamParameters
    : rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<RltoolsTypePolicy> {
    using T = typename RltoolsTypePolicy::DEFAULT;
    static constexpr T ALPHA = kLearningRate;
    static constexpr T BETA_1 = kAdamBeta1;
    static constexpr T BETA_2 = kAdamBeta2;
    static constexpr T EPSILON = kAdamEpsilon;
    static constexpr T EPSILON_SQRT = 0.0F;
};

using RltoolsCapability =
    rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, false>;
using RltoolsInputShape = rlt::tensor::Shape<RltoolsTI, 1, kInputFeatures>;

using RltoolsDense1Config = rlt::nn::layers::dense::Configuration<
    RltoolsTypePolicy,
    RltoolsTI,
    kHidden1,
    rlt::nn::activation_functions::RELU,
    rlt::nn::layers::dense::DefaultInitializer<RltoolsTypePolicy, RltoolsTI>,
    rlt::nn::parameters::groups::Input>;
using RltoolsDense2Config = rlt::nn::layers::dense::Configuration<
    RltoolsTypePolicy,
    RltoolsTI,
    kHidden2,
    rlt::nn::activation_functions::RELU,
    rlt::nn::layers::dense::DefaultInitializer<RltoolsTypePolicy, RltoolsTI>,
    rlt::nn::parameters::groups::Normal>;
using RltoolsOutputConfig = rlt::nn::layers::dense::Configuration<
    RltoolsTypePolicy,
    RltoolsTI,
    kOutputFeatures,
    rlt::nn::activation_functions::IDENTITY,
    rlt::nn::layers::dense::DefaultInitializer<RltoolsTypePolicy, RltoolsTI>,
    rlt::nn::parameters::groups::Output>;

using RltoolsDense1 = rlt::nn::layers::dense::BindConfiguration<RltoolsDense1Config>;
using RltoolsDense2 = rlt::nn::layers::dense::BindConfiguration<RltoolsDense2Config>;
using RltoolsOutput = rlt::nn::layers::dense::BindConfiguration<RltoolsOutputConfig>;

template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential::OutputModule>
using RltoolsSequentialModule = rlt::nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;

using RltoolsModuleChain =
    RltoolsSequentialModule<RltoolsDense1,
        RltoolsSequentialModule<RltoolsDense2,
            RltoolsSequentialModule<RltoolsOutput>>>;
using RltoolsModel =
    rlt::nn_models::sequential::Build<RltoolsCapability, RltoolsModuleChain, RltoolsInputShape>;
using RltoolsModelGradient =
    rlt::nn_models::sequential::ModuleGradient<typename RltoolsModel::SPEC>;
using RltoolsOptimizerSpec =
    rlt::nn::optimizers::adam::Specification<RltoolsTypePolicy, RltoolsTI, RltoolsAdamParameters, false>;
using RltoolsOptimizer = rlt::nn::optimizers::Adam<RltoolsOptimizerSpec>;
using RltoolsBuffer = typename RltoolsModel::template Buffer<false>;
using RltoolsInputTensor = rlt::Tensor<
    rlt::tensor::Specification<float, RltoolsTI, RltoolsInputShape, false>>;
using RltoolsOutputTensor = rlt::Tensor<
    rlt::tensor::Specification<float, RltoolsTI, typename RltoolsModel::OUTPUT_SHAPE, false>>;

constexpr std::size_t kRltoolsParameterCount =
    kInputFeatures * kHidden1 + kHidden1 +
    kHidden1 * kHidden2 + kHidden2 +
    kHidden2 * kOutputFeatures + kOutputFeatures;
static_assert(kRltoolsParameterCount == GenericModel::parameter_count);

struct RltoolsRuntime {
    RltoolsDevice device{};
    RltoolsModel model{};
    RltoolsOptimizer optimizer{};
    RltoolsBuffer buffers{};
    RltoolsInputTensor input{};
    RltoolsOutputTensor target{};
    RltoolsOutputTensor output_gradient{};
    RltoolsRng rng{};
};

constexpr std::size_t kRltoolsStaticStateBytes = sizeof(RltoolsRuntime);
constexpr std::size_t kRltoolsModelObjectBytes = sizeof(RltoolsModel);
#else
constexpr std::size_t kRltoolsStaticStateBytes = 0;
constexpr std::size_t kRltoolsModelObjectBytes = 0;
#endif

struct RunStats {
    std::uint32_t cycles = 0;
    std::uint32_t profile_cycles = 0;
    int status = 0;
    struct Profile {
        std::uint64_t zero_grad = 0;
        std::uint64_t input_copy = 0;
        std::uint64_t forward = 0;
        std::uint64_t loss = 0;
        std::uint64_t backward = 0;
        std::uint64_t sample_train_step = 0;
        std::uint64_t adam_update = 0;

        std::uint64_t component_sum() const noexcept {
            return zero_grad + input_copy + forward + loss + backward +
                   sample_train_step + adam_update;
        }
    } profile{};
};

struct RunAggregate {
    std::uint64_t cycles_total = 0;
    std::uint64_t profile_cycles_total = 0;
    std::uint32_t cycles_min = static_cast<std::uint32_t>(~0UL);
    std::uint32_t cycles_max = 0;
    int status = 0;
    RunStats::Profile profile_total{};
};

struct RegressionDataset {
    std::array<std::array<float, kInputFeatures>, kRolloutSamples> inputs{};
    std::array<std::array<float, kOutputFeatures>, kRolloutSamples> targets{};
};

struct TraceMeta {
    const char* family = "";
    const char* variant = "";
    std::uint32_t seed = 0;
};

constexpr std::uint32_t hash32(std::uint32_t value) noexcept {
    value ^= value >> 16U;
    value *= 0x7FEB352DU;
    value ^= value >> 15U;
    value *= 0x846CA68BU;
    value ^= value >> 16U;
    return value;
}

float signed_unit_from_hash(std::uint32_t value) noexcept {
    const std::uint32_t h = hash32(value);
    const float u01 = static_cast<float>(h & 0xFFFFU) / 65535.0F;
    return (u01 * 2.0F) - 1.0F;
}

[[maybe_unused]] float absf(float value) noexcept {
    return value < 0.0F ? -value : value;
}

template<std::size_t N>
void fill_linear_regression_sample(std::array<float, N>& input,
                                   std::array<float, 1>& target,
                                   std::size_t sample_index,
                                   std::uint32_t seed) noexcept {
    float y = 0.075F + 0.001F * static_cast<float>(seed);
    for (std::size_t i = 0; i < N; ++i) {
        const auto h = static_cast<std::uint32_t>(
            (sample_index + 1U) * 2654435761U + (i + 17U) * 2246822519U +
            (seed + 31U) * 3266489917U);
        const float x = 0.5F * signed_unit_from_hash(h);
        const int centered = static_cast<int>((i + seed) % 9U) - 4;
        const float w = static_cast<float>(centered) * 0.0125F;
        input[i] = x;
        y += w * x;
    }
    target[0] = y;
}

void fill_dataset(RegressionDataset& dataset, std::uint32_t seed) noexcept {
    for (std::size_t sample = 0; sample < kRolloutSamples; ++sample) {
        fill_linear_regression_sample(dataset.inputs[sample],
                                      dataset.targets[sample],
                                      sample,
                                      seed);
    }
}

template<typename Model>
void fill_initial_params(std::array<float, Model::parameter_count>& params,
                         std::uint32_t seed) noexcept {
    for (std::size_t i = 0; i < params.size(); ++i) {
        const auto h = static_cast<std::uint32_t>(
            (i + 1U) * 40503U + 0x9E3779B9U + (seed + 1U) * 747796405U);
        params[i] = 0.03F * signed_unit_from_hash(h);
    }
}

void uart_write(const char* text) {
    if (text == nullptr) {
        return;
    }
    std::size_t len = 0;
    while (text[len] != '\0') {
        ++len;
    }
    if (len > 0) {
        (void)HAL_UART_Transmit(
            &hlpuart1, reinterpret_cast<std::uint8_t*>(const_cast<char*>(text)),
            static_cast<std::uint16_t>(len), HAL_MAX_DELAY);
    }
}

template<typename... Args>
void uart_printf(const char* fmt, Args... args) {
    char line[1024];
    const int written = std::snprintf(line, sizeof(line), fmt, args...);
    if (written > 0) {
        uart_write(line);
    }
}

void emit_trace(const TraceMeta& trace,
                std::size_t epoch,
                std::size_t batch,
                std::size_t step,
                float minibatch_mse) {
    uart_printf("TRACE family=%s variant=%s seed=%lu config=%u-%ux%u-1 "
                "epoch=%u batch=%u step=%u sample_passes=%u minibatch_mse_e-9=%ld\r\n",
                trace.family,
                trace.variant,
                static_cast<unsigned long>(trace.seed),
                static_cast<unsigned>(kInputFeatures),
                static_cast<unsigned>(kHidden1),
                static_cast<unsigned>(kHidden2),
                static_cast<unsigned>(epoch),
                static_cast<unsigned>(batch),
                static_cast<unsigned>(step),
                static_cast<unsigned>(step * kBatchSize),
                static_cast<long>(minibatch_mse * 1000000000.0F));
}

void dwt_start() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    DWT->CYCCNT = 0U;
}

std::uint32_t dwt_read() {
    return DWT->CYCCNT;
}

void profile_add(std::uint64_t& bucket, std::uint32_t begin) {
    bucket += static_cast<std::uint32_t>(dwt_read() - begin);
}

template<typename Model>
Model& cpp_model_instance() {
    static Model model;
    return model;
}

template<typename Model>
edge::Adam& cpp_adam_instance() {
    static edge::Adam adam{kAdamConfig};
    return adam;
}

template<typename Model>
std::uint32_t& cpp_direct_c_adam_step() {
    static std::uint32_t step = 0;
    return step;
}

el_adam_config_t legacy_adam_config() {
    return el_adam_config_t{
        .beta1 = kAdamBeta1,
        .beta2 = kAdamBeta2,
        .epsilon = kAdamEpsilon,
    };
}

template<typename Model>
int prepare_cpp_static_model(const std::array<float, Model::parameter_count>& initial_params) {
    Model& model = cpp_model_instance<Model>();
    if (model.import_parameters(initial_params) != edge::Status::Ok ||
        model.zero_grad() != edge::Status::Ok ||
        model.zero_optimizer_state() != edge::Status::Ok) {
        return -1;
    }
    cpp_adam_instance<Model>() = edge::Adam{kAdamConfig};
    cpp_direct_c_adam_step<Model>() = 0U;
    return 0;
}

template<typename Model, bool Trace = false>
int train_cpp_static_batch(const RegressionDataset& dataset,
                           const TraceMeta* trace = nullptr,
                           RunStats::Profile* profile = nullptr) {
    Model& model = cpp_model_instance<Model>();
    static std::array<float, Model::output_size> output_gradient{};

    for (std::size_t epoch = 0; epoch < kEpochs; ++epoch) {
        for (std::size_t batch = 0; batch < kBatchesPerEpoch; ++batch) {
            [[maybe_unused]] float batch_loss = 0.0F;
            std::uint32_t profile_begin = dwt_read();
            if (model.zero_grad() != edge::Status::Ok) {
                return -1;
            }
            if (profile != nullptr) {
                profile_add(profile->zero_grad, profile_begin);
            }
            const std::size_t begin = batch * kBatchSize;
            for (std::size_t i = 0; i < kBatchSize; ++i) {
                const std::size_t sample = begin + i;
                profile_begin = dwt_read();
                if (model.forward(dataset.inputs[sample]) != edge::Status::Ok) {
                    return -2;
                }
                if (profile != nullptr) {
                    profile_add(profile->forward, profile_begin);
                }
                edge::TensorView<float, Model::output_size> gradient_view(output_gradient);
                profile_begin = dwt_read();
                [[maybe_unused]] const float loss = edge::MSE::evaluate(
                    model.output(),
                    edge::TensorView<const float, Model::output_size>(dataset.targets[sample]),
                    gradient_view);
                if (profile != nullptr) {
                    profile_add(profile->loss, profile_begin);
                }
                if constexpr (Trace) {
                    batch_loss += loss;
                }
                profile_begin = dwt_read();
                if (model.backward(output_gradient) != edge::Status::Ok) {
                    return -3;
                }
                if (profile != nullptr) {
                    profile_add(profile->backward, profile_begin);
                }
            }
            profile_begin = dwt_read();
            if (cpp_adam_instance<Model>().step(model, 1.0F) != edge::Status::Ok) {
                return -4;
            }
            if (profile != nullptr) {
                profile_add(profile->adam_update, profile_begin);
            }
            if constexpr (Trace) {
                if (trace != nullptr) {
                    const std::size_t step = epoch * kBatchesPerEpoch + batch + 1U;
                    emit_trace(*trace, epoch, batch, step, batch_loss / kBatchSize);
                }
            }
        }
    }
    return 0;
}

template<typename Model, bool Trace = false>
int train_cpp_direct_c_backend_static_batch(const RegressionDataset& dataset,
                                            const TraceMeta* trace = nullptr,
                                            RunStats::Profile* profile = nullptr) {
    Model& model = cpp_model_instance<Model>();
    static std::array<float, Model::output_size> output_gradient{};

    el_tensor_t params_t{
        .data = model.parameter_data(),
        .rows = static_cast<std::uint16_t>(Model::parameter_count),
        .cols = 1U,
    };
    el_tensor_t gradients_t{
        .data = model.gradient_data(),
        .rows = static_cast<std::uint16_t>(Model::parameter_count),
        .cols = 1U,
    };
    el_tensor_t m_t{
        .data = model.optimizer_state_data(),
        .rows = static_cast<std::uint16_t>(Model::parameter_count),
        .cols = 1U,
    };
    el_tensor_t v_t{
        .data = model.optimizer_state_data() + Model::parameter_count,
        .rows = static_cast<std::uint16_t>(Model::parameter_count),
        .cols = 1U,
    };
    const el_adam_config_t adam_config = legacy_adam_config();

    for (std::size_t epoch = 0; epoch < kEpochs; ++epoch) {
        for (std::size_t batch = 0; batch < kBatchesPerEpoch; ++batch) {
            [[maybe_unused]] float batch_loss = 0.0F;
            std::uint32_t profile_begin = dwt_read();
            if (model.zero_grad() != edge::Status::Ok) {
                return -1;
            }
            if (profile != nullptr) {
                profile_add(profile->zero_grad, profile_begin);
            }
            const std::size_t begin = batch * kBatchSize;
            for (std::size_t i = 0; i < kBatchSize; ++i) {
                const std::size_t sample = begin + i;
                profile_begin = dwt_read();
                if (model.forward(dataset.inputs[sample]) != edge::Status::Ok) {
                    return -2;
                }
                if (profile != nullptr) {
                    profile_add(profile->forward, profile_begin);
                }
                edge::TensorView<float, Model::output_size> gradient_view(output_gradient);
                profile_begin = dwt_read();
                [[maybe_unused]] const float loss = edge::MSE::evaluate(
                    model.output(),
                    edge::TensorView<const float, Model::output_size>(dataset.targets[sample]),
                    gradient_view);
                if (profile != nullptr) {
                    profile_add(profile->loss, profile_begin);
                }
                if constexpr (Trace) {
                    batch_loss += loss;
                }
                profile_begin = dwt_read();
                if (model.backward(output_gradient) != edge::Status::Ok) {
                    return -3;
                }
                if (profile != nullptr) {
                    profile_add(profile->backward, profile_begin);
                }
            }
            std::uint32_t& step = cpp_direct_c_adam_step<Model>();
            ++step;
            profile_begin = dwt_read();
            el_backend_adam_update_ex(
                &params_t, &gradients_t, &m_t, &v_t, kLearningRate, step, &adam_config);
            if (profile != nullptr) {
                profile_add(profile->adam_update, profile_begin);
            }
            if constexpr (Trace) {
                if (trace != nullptr) {
                    const std::size_t trace_step = epoch * kBatchesPerEpoch + batch + 1U;
                    emit_trace(*trace, epoch, batch, trace_step, batch_loss / kBatchSize);
                }
            }
        }
    }
    return 0;
}

template<typename Model>
int export_cpp_static_params(std::array<float, Model::parameter_count>& trained_params) {
    Model& model = cpp_model_instance<Model>();
    return model.export_parameters(trained_params) == edge::Status::Ok ? 0 : -4;
}

#if EL_CVSCPP_ENABLE_RLTOOLS
RltoolsRuntime& rltools_runtime() {
    static RltoolsRuntime runtime{};
    return runtime;
}

RltoolsModelGradient& rltools_model_view(RltoolsRuntime& runtime) {
    return static_cast<RltoolsModelGradient&>(runtime.model);
}

int prepare_rltools_static_model(const std::array<float, kRltoolsParameterCount>& initial_params) {
    RltoolsRuntime& runtime = rltools_runtime();
    RltoolsModelGradient& model = rltools_model_view(runtime);
    runtime.rng.state = 0;
    rlt::init(runtime.device, runtime.optimizer);
    rlt::reset_optimizer_state(runtime.device, runtime.optimizer, model);
    rlt::zero_gradient(runtime.device, model);

    std::size_t offset = 0;
    auto& layer1 = rlt::get_layer<0>(model);
    for (std::size_t out = 0; out < kHidden1; ++out) {
        for (std::size_t in = 0; in < kInputFeatures; ++in) {
            rlt::set(runtime.device, layer1.weights.parameters, initial_params[offset++], out, in);
        }
    }
    for (std::size_t out = 0; out < kHidden1; ++out) {
        rlt::set(runtime.device, layer1.biases.parameters, initial_params[offset++], out);
    }

    auto& layer2 = rlt::get_layer<1>(model);
    for (std::size_t out = 0; out < kHidden2; ++out) {
        for (std::size_t in = 0; in < kHidden1; ++in) {
            rlt::set(runtime.device, layer2.weights.parameters, initial_params[offset++], out, in);
        }
    }
    for (std::size_t out = 0; out < kHidden2; ++out) {
        rlt::set(runtime.device, layer2.biases.parameters, initial_params[offset++], out);
    }

    auto& output = rlt::get_layer<2>(model);
    for (std::size_t out = 0; out < kOutputFeatures; ++out) {
        for (std::size_t in = 0; in < kHidden2; ++in) {
            rlt::set(runtime.device, output.weights.parameters, initial_params[offset++], out, in);
        }
    }
    for (std::size_t out = 0; out < kOutputFeatures; ++out) {
        rlt::set(runtime.device, output.biases.parameters, initial_params[offset++], out);
    }
    return offset == initial_params.size() ? 0 : -1;
}

template<bool Trace = false>
int train_rltools_static_batch(const RegressionDataset& dataset,
                               const TraceMeta* trace = nullptr,
                               RunStats::Profile* profile = nullptr) {
    RltoolsRuntime& runtime = rltools_runtime();
    RltoolsModelGradient& model = rltools_model_view(runtime);

    for (std::size_t epoch = 0; epoch < kEpochs; ++epoch) {
        for (std::size_t batch = 0; batch < kBatchesPerEpoch; ++batch) {
            [[maybe_unused]] float batch_loss = 0.0F;
            std::uint32_t profile_begin = dwt_read();
            rlt::zero_gradient(runtime.device, model);
            if (profile != nullptr) {
                profile_add(profile->zero_grad, profile_begin);
            }
            const std::size_t begin = batch * kBatchSize;
            for (std::size_t i = 0; i < kBatchSize; ++i) {
                const std::size_t sample = begin + i;
                profile_begin = dwt_read();
                for (std::size_t feature = 0; feature < kInputFeatures; ++feature) {
                    rlt::set(runtime.device,
                             runtime.input,
                             dataset.inputs[sample][feature],
                             0,
                             feature);
                }
                rlt::set(runtime.device, runtime.target, dataset.targets[sample][0], 0, 0);
                if (profile != nullptr) {
                    profile_add(profile->input_copy, profile_begin);
                }

                profile_begin = dwt_read();
                rlt::forward(runtime.device,
                             model,
                             runtime.input,
                             runtime.buffers,
                             runtime.rng);
                auto output = rlt::output(runtime.device, model);
                if (profile != nullptr) {
                    profile_add(profile->forward, profile_begin);
                }
                profile_begin = dwt_read();
                if constexpr (Trace) {
                    batch_loss +=
                        rlt::nn::loss_functions::mse::evaluate(
                            runtime.device, output, runtime.target);
                }
                rlt::nn::loss_functions::mse::gradient(
                    runtime.device, output, runtime.target, runtime.output_gradient);
                if (profile != nullptr) {
                    profile_add(profile->loss, profile_begin);
                }
                profile_begin = dwt_read();
                rlt::backward(runtime.device,
                              model,
                              runtime.input,
                              runtime.output_gradient,
                              runtime.buffers);
                if (profile != nullptr) {
                    profile_add(profile->backward, profile_begin);
                }
            }
            profile_begin = dwt_read();
            rlt::step(runtime.device, runtime.optimizer, model);
            if (profile != nullptr) {
                profile_add(profile->adam_update, profile_begin);
            }
            if constexpr (Trace) {
                if (trace != nullptr) {
                    const std::size_t step = epoch * kBatchesPerEpoch + batch + 1U;
                    emit_trace(*trace, epoch, batch, step, batch_loss / kBatchSize);
                }
            }
        }
    }
    return 0;
}

int export_rltools_static_params(std::array<float, kRltoolsParameterCount>& trained_params) {
    RltoolsRuntime& runtime = rltools_runtime();
    RltoolsModelGradient& model = rltools_model_view(runtime);
    std::size_t offset = 0;

    auto& layer1 = rlt::get_layer<0>(model);
    for (std::size_t out = 0; out < kHidden1; ++out) {
        for (std::size_t in = 0; in < kInputFeatures; ++in) {
            trained_params[offset++] =
                rlt::get(runtime.device, layer1.weights.parameters, out, in);
        }
    }
    for (std::size_t out = 0; out < kHidden1; ++out) {
        trained_params[offset++] = rlt::get(runtime.device, layer1.biases.parameters, out);
    }

    auto& layer2 = rlt::get_layer<1>(model);
    for (std::size_t out = 0; out < kHidden2; ++out) {
        for (std::size_t in = 0; in < kHidden1; ++in) {
            trained_params[offset++] =
                rlt::get(runtime.device, layer2.weights.parameters, out, in);
        }
    }
    for (std::size_t out = 0; out < kHidden2; ++out) {
        trained_params[offset++] = rlt::get(runtime.device, layer2.biases.parameters, out);
    }

    auto& output = rlt::get_layer<2>(model);
    for (std::size_t out = 0; out < kOutputFeatures; ++out) {
        for (std::size_t in = 0; in < kHidden2; ++in) {
            trained_params[offset++] =
                rlt::get(runtime.device, output.weights.parameters, out, in);
        }
    }
    for (std::size_t out = 0; out < kOutputFeatures; ++out) {
        trained_params[offset++] = rlt::get(runtime.device, output.biases.parameters, out);
    }
    return offset == trained_params.size() ? 0 : -2;
}
#endif

constexpr std::size_t legacy_c_align_up(std::size_t value) noexcept {
    return (value + (EL_ALIGNMENT_BYTES - 1U)) &
           ~(static_cast<std::size_t>(EL_ALIGNMENT_BYTES) - 1U);
}

constexpr std::size_t legacy_c_alloc_head(std::size_t head, std::size_t bytes) noexcept {
    return legacy_c_align_up(head) + bytes;
}

constexpr std::size_t legacy_c_tensor_head(std::size_t head,
                                           std::size_t rows,
                                           std::size_t cols) noexcept {
    head = legacy_c_alloc_head(head, sizeof(el_tensor_t));
    return legacy_c_alloc_head(head, rows * cols * sizeof(el_float_t));
}

constexpr std::size_t legacy_c_dense_head(std::size_t head,
                                          std::size_t in_features,
                                          std::size_t out_features) noexcept {
    head = legacy_c_tensor_head(head, out_features, in_features);
    head = legacy_c_tensor_head(head, out_features, 1U);
    head = legacy_c_tensor_head(head, out_features, 1U);
    head = legacy_c_tensor_head(head, out_features, in_features);
    head = legacy_c_tensor_head(head, out_features, 1U);
    head = legacy_c_tensor_head(head, out_features, in_features);
    head = legacy_c_tensor_head(head, out_features, in_features);
    head = legacy_c_tensor_head(head, out_features, 1U);
    return legacy_c_tensor_head(head, out_features, 1U);
}

constexpr std::size_t legacy_c_required_arena_bytes() noexcept {
    std::size_t head = 0;
    head = legacy_c_alloc_head(head, 3U * sizeof(el_layer_t));
    head = legacy_c_dense_head(head, kInputFeatures, kHidden1);
    head = legacy_c_dense_head(head, kHidden1, kHidden2);
    head = legacy_c_dense_head(head, kHidden2, kOutputFeatures);

    std::size_t workspace_cap = kInputFeatures;
    workspace_cap = workspace_cap > kHidden1 ? workspace_cap : kHidden1;
    workspace_cap = workspace_cap > kHidden2 ? workspace_cap : kHidden2;
    head = legacy_c_alloc_head(head, 2U * workspace_cap * sizeof(el_float_t));

    return legacy_c_align_up(head + (EL_ALIGNMENT_BYTES * 4U));
}

constexpr std::size_t kLegacyCArenaBytes = legacy_c_required_arena_bytes();
constexpr std::size_t kLegacyCControlBytes = sizeof(el_context_t) + sizeof(el_network_t);

struct LegacyCRuntime {
    alignas(EL_ALIGNMENT_BYTES) std::array<std::uint8_t, kLegacyCArenaBytes> arena{};
    el_context_t ctx{};
    el_network_t net{};
};

LegacyCRuntime& legacy_c_runtime() {
    static LegacyCRuntime runtime{};
    return runtime;
}

template<std::size_t ParamCount>
int prepare_legacy_c_static_model(const std::array<float, ParamCount>& initial_params) {
    LegacyCRuntime& runtime = legacy_c_runtime();
    runtime.arena.fill(0U);
    if (el_ctx_init(&runtime.ctx, runtime.arena.data(),
                    static_cast<std::uint32_t>(runtime.arena.size())) != EL_OK ||
        el_network_init(&runtime.ctx, &runtime.net, 3) != EL_OK) {
        return -1;
    }
    if (el_network_add_dense(&runtime.net, static_cast<std::uint16_t>(kInputFeatures),
                             static_cast<std::uint16_t>(kHidden1), EL_ACT_RELU,
                             true, true) == nullptr ||
        el_network_add_dense(&runtime.net, static_cast<std::uint16_t>(kHidden1),
                             static_cast<std::uint16_t>(kHidden2), EL_ACT_RELU,
                             true, true) == nullptr ||
        el_network_add_dense(&runtime.net, static_cast<std::uint16_t>(kHidden2),
                             static_cast<std::uint16_t>(kOutputFeatures), EL_ACT_NONE,
                             true, true) == nullptr ||
        el_network_prepare_training(&runtime.net) != EL_OK ||
        el_network_param_count(&runtime.net) != initial_params.size() ||
        !el_network_import_params(&runtime.net, initial_params.data(), initial_params.size())) {
        return -2;
    }
    el_network_zero_grad(&runtime.net);
    return 0;
}

template<bool Trace = false>
int train_legacy_c_static_batch(const RegressionDataset& dataset,
                                const TraceMeta* trace = nullptr,
                                RunStats::Profile* profile = nullptr) {
    LegacyCRuntime& runtime = legacy_c_runtime();
    const el_adam_config_t adam_config = legacy_adam_config();

    for (std::size_t epoch = 0; epoch < kEpochs; ++epoch) {
        for (std::size_t batch = 0; batch < kBatchesPerEpoch; ++batch) {
            [[maybe_unused]] float batch_loss = 0.0F;
            std::uint32_t profile_begin = dwt_read();
            el_network_zero_grad(&runtime.net);
            if (profile != nullptr) {
                profile_add(profile->zero_grad, profile_begin);
            }
            const std::size_t begin = batch * kBatchSize;
            for (std::size_t i = 0; i < kBatchSize; ++i) {
                const std::size_t sample = begin + i;
                el_tensor_t sample_input{
                    .data = const_cast<float*>(dataset.inputs[sample].data()),
                    .rows = static_cast<std::uint16_t>(kInputFeatures),
                    .cols = 1U,
                };
                el_tensor_t sample_target{
                    .data = const_cast<float*>(dataset.targets[sample].data()),
                    .rows = 1U,
                    .cols = 1U,
                };
                profile_begin = dwt_read();
                [[maybe_unused]] const float loss = el_network_train_step(
                    &runtime.net, &sample_input, &sample_target, EL_LOSS_MSE);
                if (profile != nullptr) {
                    profile_add(profile->sample_train_step, profile_begin);
                }
                if constexpr (Trace) {
                    batch_loss += loss;
                }
            }
            profile_begin = dwt_read();
            el_network_update_with_adam_config(&runtime.net, kLearningRate, &adam_config);
            if (profile != nullptr) {
                profile_add(profile->adam_update, profile_begin);
            }
            if constexpr (Trace) {
                if (trace != nullptr) {
                    const std::size_t step = epoch * kBatchesPerEpoch + batch + 1U;
                    emit_trace(*trace, epoch, batch, step, batch_loss / kBatchSize);
                }
            }
        }
    }
    return 0;
}

template<std::size_t ParamCount>
int export_legacy_c_static_params(std::array<float, ParamCount>& trained_params) {
    LegacyCRuntime& runtime = legacy_c_runtime();
    return el_network_export_params(&runtime.net, trained_params.data(), trained_params.size())
               ? 0
               : -3;
}

template<typename Fn>
RunStats timed(Fn&& fn) {
    dwt_start();
    const std::uint32_t begin = dwt_read();
    const int status = fn();
    const std::uint32_t end = dwt_read();
    return RunStats{.cycles = end - begin, .status = status};
}

void aggregate_update(RunAggregate& aggregate, RunStats stats) {
    aggregate.cycles_total += stats.cycles;
    aggregate.profile_cycles_total += stats.profile_cycles;
    aggregate.profile_total.zero_grad += stats.profile.zero_grad;
    aggregate.profile_total.input_copy += stats.profile.input_copy;
    aggregate.profile_total.forward += stats.profile.forward;
    aggregate.profile_total.loss += stats.profile.loss;
    aggregate.profile_total.backward += stats.profile.backward;
    aggregate.profile_total.sample_train_step += stats.profile.sample_train_step;
    aggregate.profile_total.adam_update += stats.profile.adam_update;
    if (stats.cycles < aggregate.cycles_min) {
        aggregate.cycles_min = stats.cycles;
    }
    if (stats.cycles > aggregate.cycles_max) {
        aggregate.cycles_max = stats.cycles;
    }
    if (stats.status != 0 && aggregate.status == 0) {
        aggregate.status = stats.status;
    }
}

std::uint64_t profile_gap(std::uint64_t cycles, const RunStats::Profile& profile) {
    const std::uint64_t component_sum = profile.component_sum();
    return cycles > component_sum ? cycles - component_sum : 0U;
}

template<std::size_t N>
float max_abs_diff(const std::array<float, N>& lhs, const std::array<float, N>& rhs) {
    float max_diff = 0.0F;
    for (std::size_t i = 0; i < N; ++i) {
        const float diff = absf(lhs[i] - rhs[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

template<std::size_t N>
int emit_compare(const char* name,
                 std::uint32_t seed,
                 const std::array<float, N>& lhs,
                 const std::array<float, N>& rhs) {
    const float diff = max_abs_diff(lhs, rhs);
    const int ok = diff <= kTolerance ? 1 : 0;
    uart_printf("COMPARE seed=%lu name=%s max_abs_diff_e-9=%ld tolerance_e-9=%ld ok=%d\r\n",
                static_cast<unsigned long>(seed),
                name,
                static_cast<long>(diff * 1000000000.0F),
                static_cast<long>(kTolerance * 1000000000.0F),
                ok);
    return ok ? 0 : -1;
}

void emit_result(const char* family,
                 const char* variant,
                 std::uint32_t seed,
                 RunStats stats,
                 std::size_t arena_bytes,
                 std::size_t object_bytes) {
    uart_printf("RESULT family=%s variant=%s seed=%lu config=%u-%ux%u-1 batch=%u "
                "rollout=%u epochs=%u optimizer_steps=%u sample_passes=%u warmups=%u "
                "params=%u cycles=%lu prof_cycles=%lu status=%d arena_bytes=%u object_bytes=%u "
                "prof_zero=%lu prof_input_copy=%lu prof_forward=%lu prof_loss=%lu "
                "prof_backward=%lu prof_sample_train_step=%lu prof_adam_update=%lu "
                "prof_component_sum=%lu prof_gap=%lu\r\n",
                family,
                variant,
                static_cast<unsigned long>(seed),
                static_cast<unsigned>(kInputFeatures),
                static_cast<unsigned>(kHidden1),
                static_cast<unsigned>(kHidden2),
                static_cast<unsigned>(kBatchSize),
                static_cast<unsigned>(kRolloutSamples),
                static_cast<unsigned>(kEpochs),
                static_cast<unsigned>(kOptimizerSteps),
                static_cast<unsigned>(kSamplePasses),
                static_cast<unsigned>(kWarmupRuns),
                static_cast<unsigned>(M55Model::parameter_count),
                static_cast<unsigned long>(stats.cycles),
                static_cast<unsigned long>(stats.profile_cycles),
                stats.status,
                static_cast<unsigned>(arena_bytes),
                static_cast<unsigned>(object_bytes),
                static_cast<unsigned long>(stats.profile.zero_grad),
                static_cast<unsigned long>(stats.profile.input_copy),
                static_cast<unsigned long>(stats.profile.forward),
                static_cast<unsigned long>(stats.profile.loss),
                static_cast<unsigned long>(stats.profile.backward),
                static_cast<unsigned long>(stats.profile.sample_train_step),
                static_cast<unsigned long>(stats.profile.adam_update),
                static_cast<unsigned long>(stats.profile.component_sum()),
                static_cast<unsigned long>(profile_gap(stats.profile_cycles, stats.profile)));
}

void emit_summary(const char* family,
                  const char* variant,
                  const RunAggregate& aggregate,
                  std::size_t arena_bytes,
                  std::size_t object_bytes) {
    const std::uint64_t avg = aggregate.cycles_total / kSeedCount;
    const std::uint64_t profile_avg_cycles = aggregate.profile_cycles_total / kSeedCount;
    RunStats::Profile profile_avg{};
    profile_avg.zero_grad = aggregate.profile_total.zero_grad / kSeedCount;
    profile_avg.input_copy = aggregate.profile_total.input_copy / kSeedCount;
    profile_avg.forward = aggregate.profile_total.forward / kSeedCount;
    profile_avg.loss = aggregate.profile_total.loss / kSeedCount;
    profile_avg.backward = aggregate.profile_total.backward / kSeedCount;
    profile_avg.sample_train_step = aggregate.profile_total.sample_train_step / kSeedCount;
    profile_avg.adam_update = aggregate.profile_total.adam_update / kSeedCount;
    uart_printf("SUMMARY family=%s variant=%s config=%u-%ux%u-1 seeds=%u batch=%u "
                "rollout=%u epochs=%u optimizer_steps=%u sample_passes=%u warmups=%u params=%u "
                "cycles_avg=%lu cycles_min=%lu cycles_max=%lu prof_cycles_avg=%lu status=%d "
                "arena_bytes=%u object_bytes=%u "
                "prof_zero_avg=%lu prof_input_copy_avg=%lu prof_forward_avg=%lu "
                "prof_loss_avg=%lu prof_backward_avg=%lu prof_sample_train_step_avg=%lu "
                "prof_adam_update_avg=%lu prof_component_sum_avg=%lu prof_gap_avg=%lu\r\n",
                family,
                variant,
                static_cast<unsigned>(kInputFeatures),
                static_cast<unsigned>(kHidden1),
                static_cast<unsigned>(kHidden2),
                static_cast<unsigned>(kSeedCount),
                static_cast<unsigned>(kBatchSize),
                static_cast<unsigned>(kRolloutSamples),
                static_cast<unsigned>(kEpochs),
                static_cast<unsigned>(kOptimizerSteps),
                static_cast<unsigned>(kSamplePasses),
                static_cast<unsigned>(kWarmupRuns),
                static_cast<unsigned>(M55Model::parameter_count),
                static_cast<unsigned long>(avg),
                static_cast<unsigned long>(aggregate.cycles_min),
                static_cast<unsigned long>(aggregate.cycles_max),
                static_cast<unsigned long>(profile_avg_cycles),
                aggregate.status,
                static_cast<unsigned>(arena_bytes),
                static_cast<unsigned>(object_bytes),
                static_cast<unsigned long>(profile_avg.zero_grad),
                static_cast<unsigned long>(profile_avg.input_copy),
                static_cast<unsigned long>(profile_avg.forward),
                static_cast<unsigned long>(profile_avg.loss),
                static_cast<unsigned long>(profile_avg.backward),
                static_cast<unsigned long>(profile_avg.sample_train_step),
                static_cast<unsigned long>(profile_avg.adam_update),
                static_cast<unsigned long>(profile_avg.component_sum()),
                static_cast<unsigned long>(profile_gap(profile_avg_cycles, profile_avg)));
}

void emit_begin() {
    uart_printf("BEGIN test=EL_C_vsCpp variant=%s config=%u-%ux%u-1 seeds=%u batch=%u "
                "rollout=%u epochs=%u optimizer_steps=%u sample_passes=%u warmups=%u "
                "trace_seed=%lu input_features=%u params=%u "
                "legacy_c_arena=%u legacy_c_control=%u "
                "cpp_direct_required_memory=%u cpp_direct_model_object=%u "
                "cpp_m55_required_memory=%u cpp_m55_model_object=%u "
                "cpp_generic_required_memory=%u cpp_generic_model_object=%u "
                "rltools_static_state=%u rltools_model_object=%u "
                "optimizer=adam lr_e-9=%ld beta1_e-9=%ld beta2_e-9=%ld eps_e-12=%ld "
                "timing=ppo_like_sample_passes profile_schema=train_loop_components_v1 "
                "mve=%d opt=Ofast\r\n",
                kVariantName,
                static_cast<unsigned>(kInputFeatures),
                static_cast<unsigned>(kHidden1),
                static_cast<unsigned>(kHidden2),
                static_cast<unsigned>(kSeedCount),
                static_cast<unsigned>(kBatchSize),
                static_cast<unsigned>(kRolloutSamples),
                static_cast<unsigned>(kEpochs),
                static_cast<unsigned>(kOptimizerSteps),
                static_cast<unsigned>(kSamplePasses),
                static_cast<unsigned>(kWarmupRuns),
                static_cast<unsigned long>(kConvergenceTraceSeed),
                static_cast<unsigned>(kInputFeatures),
                static_cast<unsigned>(M55Model::parameter_count),
                static_cast<unsigned>(kLegacyCArenaBytes),
                static_cast<unsigned>(kLegacyCControlBytes),
                static_cast<unsigned>(DirectCBackendModel::required_memory),
                static_cast<unsigned>(sizeof(DirectCBackendModel)),
                static_cast<unsigned>(M55Model::required_memory),
                static_cast<unsigned>(sizeof(M55Model)),
                static_cast<unsigned>(GenericModel::required_memory),
                static_cast<unsigned>(sizeof(GenericModel)),
                static_cast<unsigned>(kRltoolsStaticStateBytes),
                static_cast<unsigned>(kRltoolsModelObjectBytes),
                static_cast<long>(kLearningRate * 1000000000.0F),
                static_cast<long>(kAdamBeta1 * 1000000000.0F),
                static_cast<long>(kAdamBeta2 * 1000000000.0F),
                static_cast<long>(kAdamEpsilon * 1000000000000.0F),
                edge::detail::m55_mve_available ? 1 : 0);
}

template<std::size_t ParamCount>
RunStats run_legacy_c_once(const std::array<float, ParamCount>& initial_params,
                           const RegressionDataset& dataset,
                           std::array<float, ParamCount>& trained_params,
                           std::uint32_t seed) {
    for (std::size_t warmup = 0; warmup < kWarmupRuns; ++warmup) {
        const int prepare_status = prepare_legacy_c_static_model(initial_params);
        if (prepare_status != 0) {
            return RunStats{.cycles = 0, .status = prepare_status};
        }
        const int warmup_status = train_legacy_c_static_batch<false>(dataset);
        if (warmup_status != 0) {
            return RunStats{.cycles = 0, .status = warmup_status};
        }
    }

    const int prepare_status = prepare_legacy_c_static_model(initial_params);
    RunStats stats = prepare_status == 0
        ? timed([&]() { return train_legacy_c_static_batch<false>(dataset); })
        : RunStats{.cycles = 0, .status = prepare_status};
    if (stats.status == 0) {
        stats.status = export_legacy_c_static_params(trained_params);
    }
    if (stats.status == 0) {
        RunStats::Profile profile{};
        const int profile_prepare_status = prepare_legacy_c_static_model(initial_params);
        const RunStats profile_stats = profile_prepare_status == 0
            ? timed([&]() { return train_legacy_c_static_batch<false>(dataset, nullptr, &profile); })
            : RunStats{.cycles = 0, .status = profile_prepare_status};
        if (profile_stats.status != 0) {
            stats.status = -11;
        } else {
            stats.profile_cycles = profile_stats.cycles;
            stats.profile = profile;
        }
    }
    if (stats.status == 0 && seed == kConvergenceTraceSeed) {
        const TraceMeta trace{"baseline_c_m55", "legacy_c", seed};
        const int trace_prepare_status = prepare_legacy_c_static_model(initial_params);
        const int trace_status = trace_prepare_status == 0
            ? train_legacy_c_static_batch<true>(dataset, &trace)
            : trace_prepare_status;
        if (trace_status != 0) {
            stats.status = -10;
        }
    }
    return stats;
}

template<typename Model>
RunStats run_cpp_direct_c_backend_once(const std::array<float, Model::parameter_count>& initial_params,
                                       const RegressionDataset& dataset,
                                       std::array<float, Model::parameter_count>& trained_params,
                                       std::uint32_t seed) {
    for (std::size_t warmup = 0; warmup < kWarmupRuns; ++warmup) {
        const int prepare_status = prepare_cpp_static_model<Model>(initial_params);
        if (prepare_status != 0) {
            return RunStats{.cycles = 0, .status = prepare_status};
        }
        const int warmup_status =
            train_cpp_direct_c_backend_static_batch<Model, false>(dataset);
        if (warmup_status != 0) {
            return RunStats{.cycles = 0, .status = warmup_status};
        }
    }

    const int prepare_status = prepare_cpp_static_model<Model>(initial_params);
    RunStats stats = prepare_status == 0
        ? timed([&]() { return train_cpp_direct_c_backend_static_batch<Model, false>(dataset); })
        : RunStats{.cycles = 0, .status = prepare_status};
    if (stats.status == 0) {
        stats.status = export_cpp_static_params<Model>(trained_params);
    }
    if (stats.status == 0) {
        RunStats::Profile profile{};
        const int profile_prepare_status = prepare_cpp_static_model<Model>(initial_params);
        const RunStats profile_stats = profile_prepare_status == 0
            ? timed([&]() {
                  return train_cpp_direct_c_backend_static_batch<Model, false>(
                      dataset, nullptr, &profile);
              })
            : RunStats{.cycles = 0, .status = profile_prepare_status};
        if (profile_stats.status != 0) {
            stats.status = -11;
        } else {
            stats.profile_cycles = profile_stats.cycles;
            stats.profile = profile;
        }
    }
    if (stats.status == 0 && seed == kConvergenceTraceSeed) {
        const TraceMeta trace{"same_c_backend", "cpp_direct_c_backend", seed};
        const int trace_prepare_status = prepare_cpp_static_model<Model>(initial_params);
        const int trace_status = trace_prepare_status == 0
            ? train_cpp_direct_c_backend_static_batch<Model, true>(dataset, &trace)
            : trace_prepare_status;
        if (trace_status != 0) {
            stats.status = -10;
        }
    }
    return stats;
}

template<typename Model>
RunStats run_cpp_native_once(const std::array<float, Model::parameter_count>& initial_params,
                             const RegressionDataset& dataset,
                             std::array<float, Model::parameter_count>& trained_params,
                             std::uint32_t seed,
                             const char* family,
                             const char* variant) {
    for (std::size_t warmup = 0; warmup < kWarmupRuns; ++warmup) {
        const int prepare_status = prepare_cpp_static_model<Model>(initial_params);
        if (prepare_status != 0) {
            return RunStats{.cycles = 0, .status = prepare_status};
        }
        const int warmup_status = train_cpp_static_batch<Model, false>(dataset);
        if (warmup_status != 0) {
            return RunStats{.cycles = 0, .status = warmup_status};
        }
    }

    const int prepare_status = prepare_cpp_static_model<Model>(initial_params);
    RunStats stats = prepare_status == 0
        ? timed([&]() { return train_cpp_static_batch<Model, false>(dataset); })
        : RunStats{.cycles = 0, .status = prepare_status};
    if (stats.status == 0) {
        stats.status = export_cpp_static_params<Model>(trained_params);
    }
    if (stats.status == 0) {
        RunStats::Profile profile{};
        const int profile_prepare_status = prepare_cpp_static_model<Model>(initial_params);
        const RunStats profile_stats = profile_prepare_status == 0
            ? timed([&]() { return train_cpp_static_batch<Model, false>(dataset, nullptr, &profile); })
            : RunStats{.cycles = 0, .status = profile_prepare_status};
        if (profile_stats.status != 0) {
            stats.status = -11;
        } else {
            stats.profile_cycles = profile_stats.cycles;
            stats.profile = profile;
        }
    }
    if (stats.status == 0 && seed == kConvergenceTraceSeed) {
        const TraceMeta trace{family, variant, seed};
        const int trace_prepare_status = prepare_cpp_static_model<Model>(initial_params);
        const int trace_status = trace_prepare_status == 0
            ? train_cpp_static_batch<Model, true>(dataset, &trace)
            : trace_prepare_status;
        if (trace_status != 0) {
            stats.status = -10;
        }
    }
    return stats;
}

#if EL_CVSCPP_ENABLE_RLTOOLS
RunStats run_rltools_generic_once(const std::array<float, kRltoolsParameterCount>& initial_params,
                                  const RegressionDataset& dataset,
                                  std::array<float, kRltoolsParameterCount>& trained_params,
                                  std::uint32_t seed) {
    for (std::size_t warmup = 0; warmup < kWarmupRuns; ++warmup) {
        const int prepare_status = prepare_rltools_static_model(initial_params);
        if (prepare_status != 0) {
            return RunStats{.cycles = 0, .status = prepare_status};
        }
        const int warmup_status = train_rltools_static_batch<false>(dataset);
        if (warmup_status != 0) {
            return RunStats{.cycles = 0, .status = warmup_status};
        }
    }

    const int prepare_status = prepare_rltools_static_model(initial_params);
    RunStats stats = prepare_status == 0
        ? timed([&]() { return train_rltools_static_batch<false>(dataset); })
        : RunStats{.cycles = 0, .status = prepare_status};
    if (stats.status == 0) {
        stats.status = export_rltools_static_params(trained_params);
    }
    if (stats.status == 0) {
        RunStats::Profile profile{};
        const int profile_prepare_status = prepare_rltools_static_model(initial_params);
        const RunStats profile_stats = profile_prepare_status == 0
            ? timed([&]() { return train_rltools_static_batch<false>(dataset, nullptr, &profile); })
            : RunStats{.cycles = 0, .status = profile_prepare_status};
        if (profile_stats.status != 0) {
            stats.status = -11;
        } else {
            stats.profile_cycles = profile_stats.cycles;
            stats.profile = profile;
        }
    }
    if (stats.status == 0 && seed == kConvergenceTraceSeed) {
        const TraceMeta trace{"rltools_cpp_generic", "rltools_generic", seed};
        const int trace_prepare_status = prepare_rltools_static_model(initial_params);
        const int trace_status = trace_prepare_status == 0
            ? train_rltools_static_batch<true>(dataset, &trace)
            : trace_prepare_status;
        if (trace_status != 0) {
            stats.status = -10;
        }
    }
    return stats;
}
#endif

} // namespace

int el_cvscpp_ablation_run(void) {
    static std::array<float, M55Model::parameter_count> initial_params{};
#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_LEGACY_C
    static std::array<float, M55Model::parameter_count> legacy_c_params{};
#endif
#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_DIRECT_C_BACKEND
    static std::array<float, DirectCBackendModel::parameter_count> cpp_direct_c_params{};
#endif
#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_M55
    static std::array<float, M55Model::parameter_count> cpp_m55_params{};
#endif
#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_GENERIC
    static std::array<float, GenericModel::parameter_count> cpp_generic_params{};
#endif
#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_RLTOOLS_GENERIC
    static std::array<float, kRltoolsParameterCount> rltools_generic_params{};
#endif
    static RegressionDataset dataset{};

#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_LEGACY_C
    RunAggregate legacy_c_aggregate{};
#endif
#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_DIRECT_C_BACKEND
    RunAggregate cpp_direct_c_aggregate{};
#endif
#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_M55
    RunAggregate cpp_m55_aggregate{};
#endif
#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_GENERIC
    RunAggregate cpp_generic_aggregate{};
#endif
#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_RLTOOLS_GENERIC
    RunAggregate rltools_generic_aggregate{};
#endif

    emit_begin();

    int status = 0;

    for (std::uint32_t seed = 0; seed < kSeedCount; ++seed) {
        fill_initial_params<M55Model>(initial_params, seed);
        fill_dataset(dataset, seed);

#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_LEGACY_C
        RunStats legacy_c = run_legacy_c_once(initial_params, dataset, legacy_c_params, seed);
        aggregate_update(legacy_c_aggregate, legacy_c);
        emit_result("baseline_c_m55", "legacy_c", seed, legacy_c,
                    kLegacyCArenaBytes, kLegacyCControlBytes);
        if (legacy_c.status != 0 && status == 0) {
            status = -1;
        }
#endif

#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_DIRECT_C_BACKEND
        RunStats cpp_direct_c =
            run_cpp_direct_c_backend_once<DirectCBackendModel>(
                initial_params, dataset, cpp_direct_c_params, seed);
        aggregate_update(cpp_direct_c_aggregate, cpp_direct_c);
        emit_result("same_c_backend", "cpp_direct_c_backend", seed, cpp_direct_c,
                    DirectCBackendModel::required_memory, sizeof(DirectCBackendModel));
        if (cpp_direct_c.status != 0 && status == 0) {
            status = -1;
        }
#endif

#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_M55
        RunStats cpp_m55 =
            run_cpp_native_once<M55Model>(
                initial_params, dataset, cpp_m55_params, seed, "native_cpp_m55", "cpp_m55");
        aggregate_update(cpp_m55_aggregate, cpp_m55);
        emit_result("native_cpp_m55", "cpp_m55", seed, cpp_m55,
                    M55Model::required_memory, sizeof(M55Model));
        if (cpp_m55.status != 0 && status == 0) {
            status = -1;
        }
#endif

#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_GENERIC
        RunStats cpp_generic =
            run_cpp_native_once<GenericModel>(
                initial_params,
                dataset,
                cpp_generic_params,
                seed,
                "native_cpp_generic",
                "cpp_generic");
        aggregate_update(cpp_generic_aggregate, cpp_generic);
        emit_result("native_cpp_generic", "cpp_generic", seed, cpp_generic,
                    GenericModel::required_memory, sizeof(GenericModel));
        if (cpp_generic.status != 0 && status == 0) {
            status = -1;
        }
#endif

#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_RLTOOLS_GENERIC
        RunStats rltools_generic =
            run_rltools_generic_once(initial_params, dataset, rltools_generic_params, seed);
        aggregate_update(rltools_generic_aggregate, rltools_generic);
        emit_result("rltools_cpp_generic", "rltools_generic", seed, rltools_generic,
                    kRltoolsStaticStateBytes, kRltoolsModelObjectBytes);
        if (rltools_generic.status != 0 && status == 0) {
            status = -1;
        }
#endif

#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL
        if (legacy_c.status != 0 || cpp_direct_c.status != 0 ||
            cpp_m55.status != 0 || cpp_generic.status != 0 ||
            rltools_generic.status != 0) {
            status = -1;
        } else {
            if (emit_compare("legacy_c_vs_cpp_direct_c_backend",
                             seed,
                             legacy_c_params,
                             cpp_direct_c_params) != 0) {
                status = -2;
            }
            if (emit_compare("legacy_c_vs_cpp_m55",
                             seed,
                             legacy_c_params,
                             cpp_m55_params) != 0) {
                status = -3;
            }
            if (emit_compare("legacy_c_vs_cpp_generic",
                             seed,
                             legacy_c_params,
                             cpp_generic_params) != 0) {
                status = -4;
            }
            if (emit_compare("cpp_generic_vs_rltools_generic",
                             seed,
                             cpp_generic_params,
                             rltools_generic_params) != 0) {
                status = -5;
            }
        }
#endif
    }

#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_LEGACY_C
    emit_summary("baseline_c_m55", "legacy_c", legacy_c_aggregate,
                 kLegacyCArenaBytes, kLegacyCControlBytes);
#endif
#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_DIRECT_C_BACKEND
    emit_summary("same_c_backend", "cpp_direct_c_backend", cpp_direct_c_aggregate,
                 DirectCBackendModel::required_memory, sizeof(DirectCBackendModel));
#endif
#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_M55
    emit_summary("native_cpp_m55", "cpp_m55", cpp_m55_aggregate,
                 M55Model::required_memory, sizeof(M55Model));
#endif
#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_CPP_GENERIC
    emit_summary("native_cpp_generic", "cpp_generic", cpp_generic_aggregate,
                 GenericModel::required_memory, sizeof(GenericModel));
#endif
#if EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_ALL || \
    EL_CVSCPP_VARIANT == EL_CVSCPP_VARIANT_RLTOOLS_GENERIC
    emit_summary("rltools_cpp_generic", "rltools_generic", rltools_generic_aggregate,
                 kRltoolsStaticStateBytes, kRltoolsModelObjectBytes);
#endif

    uart_printf("DONE test=EL_C_vsCpp variant=%s config=%u-%ux%u-1 seeds=%u status=%d\r\n",
                kVariantName,
                static_cast<unsigned>(kInputFeatures),
                static_cast<unsigned>(kHidden1),
                static_cast<unsigned>(kHidden2),
                static_cast<unsigned>(kSeedCount),
                status);
    if (status == 0) {
        while (true) {
            __WFI();
        }
    }
    return status;
}
