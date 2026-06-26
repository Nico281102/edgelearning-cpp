#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include <edge/edge.hpp>

#include "test_harness.hpp"

#ifndef EDGE_M55_SWEEP_H1
#error "EDGE_M55_SWEEP_H1 must be defined by the build target"
#endif

#ifndef EDGE_M55_SWEEP_H2
#error "EDGE_M55_SWEEP_H2 must be defined by the build target"
#endif

#ifndef EDGE_M55_SWEEP_ENABLE_LEGACY_C
#define EDGE_M55_SWEEP_ENABLE_LEGACY_C 0
#endif

#ifndef EDGE_M55_SWEEP_ENABLE_CPP_GENERIC
#define EDGE_M55_SWEEP_ENABLE_CPP_GENERIC 1
#endif

#ifndef EDGE_M55_SWEEP_ENABLE_CPP_M55
#define EDGE_M55_SWEEP_ENABLE_CPP_M55 1
#endif

#ifndef EDGE_M55_SWEEP_ENABLE_DIRECT_C_BACKEND
#define EDGE_M55_SWEEP_ENABLE_DIRECT_C_BACKEND 0
#endif

#ifndef EDGE_M55_SWEEP_TOLERANCE
#define EDGE_M55_SWEEP_TOLERANCE 1.0e-3F
#endif

#define EDGE_M55_SWEEP_STR_HELPER(value) #value
#define EDGE_M55_SWEEP_STR(value) EDGE_M55_SWEEP_STR_HELPER(value)

#if EDGE_M55_SWEEP_ENABLE_LEGACY_C
extern "C" {
#include "edgelearning.h"
}
#endif

namespace {

constexpr std::size_t kInputFeatures = 32;
constexpr std::size_t kHidden1 = EDGE_M55_SWEEP_H1;
constexpr std::size_t kHidden2 = EDGE_M55_SWEEP_H2;
constexpr std::size_t kBatchSize = 256;
constexpr std::size_t kRolloutSamples = 1024;
constexpr std::size_t kEpochs = 2;
static_assert((kRolloutSamples % kBatchSize) == 0U);
constexpr std::size_t kBatchesPerEpoch = kRolloutSamples / kBatchSize;
constexpr std::size_t kOptimizerSteps = kEpochs * kBatchesPerEpoch;
constexpr std::size_t kSamplePasses = kEpochs * kRolloutSamples;
constexpr std::size_t kSeedCount = 10;
constexpr float kLearningRate = 1.0e-3F;
constexpr float kAdamBeta1 = 0.9F;
constexpr float kAdamBeta2 = 0.999F;
constexpr float kAdamEpsilon = 1.0e-8F;
constexpr float kTolerance = EDGE_M55_SWEEP_TOLERANCE;
constexpr edge::AdamConfig kAdamConfig{
    .learning_rate = kLearningRate,
    .beta1 = kAdamBeta1,
    .beta2 = kAdamBeta2,
    .epsilon = kAdamEpsilon,
};

using GenericModel = edge::Model<
    edge::Backend::Generic,
    edge::Input<kInputFeatures>,
    edge::Dense<kHidden1, edge::ReLU>,
    edge::Dense<kHidden2, edge::ReLU>,
    edge::Dense<1, edge::Linear>>;

using M55Model = edge::Model<
    edge::Backend::M55,
    edge::Input<kInputFeatures>,
    edge::Dense<kHidden1, edge::ReLU>,
    edge::Dense<kHidden2, edge::ReLU>,
    edge::Dense<1, edge::Linear>>;

static_assert(GenericModel::parameter_count == M55Model::parameter_count);
static_assert(GenericModel::input_size == kInputFeatures);
static_assert(GenericModel::output_size == 1U);

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

template<typename Model>
void fill_initial_params(std::array<float, Model::parameter_count>& params,
                         std::uint32_t seed) noexcept {
    for (std::size_t i = 0; i < params.size(); ++i) {
        const auto h = static_cast<std::uint32_t>(
            (i + 1U) * 40503U + 0x9E3779B9U + (seed + 1U) * 747796405U);
        params[i] = 0.03F * signed_unit_from_hash(h);
    }
}

template<std::size_t N>
void expect_all_near(const char* case_name,
                     const char* actual_name,
                     const std::array<float, N>& actual,
                     const char* expected_name,
                     const std::array<float, N>& expected,
                     float tolerance) {
    for (std::size_t i = 0; i < N; ++i) {
        const float a = actual[i];
        const float e = expected[i];
        if (!std::isfinite(a) || !std::isfinite(e) || std::fabs(a - e) > tolerance) {
            std::fprintf(stderr,
                         "%s mismatch: %s[%zu]=%.9g %s[%zu]=%.9g diff=%.9g tol=%.9g\n",
                         case_name,
                         actual_name,
                         i,
                         static_cast<double>(a),
                         expected_name,
                         i,
                         static_cast<double>(e),
                         static_cast<double>(std::fabs(a - e)),
                         static_cast<double>(tolerance));
            std::abort();
        }
    }
}

template<typename Model>
void run_cpp_static_batch(const std::array<float, Model::parameter_count>& initial_params,
                          std::array<float, Model::parameter_count>& trained_params,
                          std::uint32_t seed) {
    static Model model;
    static std::array<float, Model::input_size> input{};
    static std::array<float, Model::output_size> target{};
    static std::array<float, Model::output_size> output_gradient{};

    EDGE_EXPECT_EQ(model.import_parameters(initial_params), edge::Status::Ok);
    EDGE_EXPECT_EQ(model.zero_grad(), edge::Status::Ok);
    EDGE_EXPECT_EQ(model.zero_optimizer_state(), edge::Status::Ok);

    edge::Adam adam{kAdamConfig};
    for (std::size_t epoch = 0; epoch < kEpochs; ++epoch) {
        for (std::size_t batch = 0; batch < kBatchesPerEpoch; ++batch) {
            EDGE_EXPECT_EQ(model.zero_grad(), edge::Status::Ok);
            const std::size_t begin = batch * kBatchSize;
            for (std::size_t i = 0; i < kBatchSize; ++i) {
                const std::size_t sample = begin + i;
                fill_linear_regression_sample(input, target, sample, seed);
                EDGE_EXPECT_EQ(model.forward(input), edge::Status::Ok);
                edge::TensorView<float, Model::output_size> gradient_view(output_gradient);
                edge::MSE::evaluate(
                    model.output(),
                    edge::TensorView<const float, Model::output_size>(target),
                    gradient_view);
                EDGE_EXPECT_EQ(model.backward(output_gradient), edge::Status::Ok);
            }
            EDGE_EXPECT_EQ(adam.step(model, 1.0F), edge::Status::Ok);
        }
    }
    EDGE_EXPECT_EQ(model.export_parameters(trained_params), edge::Status::Ok);
}

#if EDGE_M55_SWEEP_ENABLE_LEGACY_C
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
            layer.activation = std::is_same_v<Activation, LegacyCReLU> ? EL_ACT_RELU : EL_ACT_NONE;

            if constexpr (std::is_same_v<Activation, LegacyCReLU>) {
                el_backend_dense_forward_relu(&input_t, &layer, &output_t);
            } else {
                el_backend_dense_forward_linear(&input_t, &layer, &output_t);
            }
        }

        template<bool PropagateInputGradient, typename Types>
        static void backward(
            edge::TensorView<const typename Types::ActivationT, in_features> input,
            edge::TensorView<const typename Types::ActivationT, out_features> output,
            edge::TensorView<const typename Types::AccumulatorT, out_features> upstream,
            edge::TensorView<typename Types::AccumulatorT,
                             PropagateInputGradient ? in_features : 0U> downstream,
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
            layer.activation = std::is_same_v<Activation, LegacyCReLU> ? EL_ACT_RELU : EL_ACT_NONE;
            layer.grad_weights = &grad_weights_t;
            layer.grad_bias = &grad_bias_t;

            if constexpr (PropagateInputGradient) {
                el_tensor_t downstream_t{
                    .data = downstream.data(),
                    .rows = static_cast<std::uint16_t>(in_features),
                    .cols = 1U,
                };
                if constexpr (std::is_same_v<Activation, LegacyCReLU>) {
                    el_backend_dense_backward_relu(
                        &layer,
                        &input_t,
                        &upstream_t,
                        &downstream_t,
                        &grad_weights_t,
                        &grad_bias_t);
                } else {
                    el_backend_dense_backward_linear(
                        &layer,
                        &input_t,
                        &upstream_t,
                        &downstream_t,
                        &grad_weights_t,
                        &grad_bias_t);
                }
            } else {
                if constexpr (std::is_same_v<Activation, LegacyCReLU>) {
                    el_backend_dense_backward_relu(
                        &layer, &input_t, &upstream_t, nullptr, &grad_weights_t, &grad_bias_t);
                } else {
                    el_backend_dense_backward_linear(
                        &layer, &input_t, &upstream_t, nullptr, &grad_weights_t, &grad_bias_t);
                }
            }
        }
    };
};

using DirectCBackendModel = edge::Model<
    edge::Backend::M55,
    edge::Input<kInputFeatures>,
    LegacyCDense<kHidden1, LegacyCReLU>,
    LegacyCDense<kHidden2, LegacyCReLU>,
    LegacyCDense<1, LegacyCLinear>>;

static_assert(DirectCBackendModel::parameter_count == GenericModel::parameter_count);
static_assert(DirectCBackendModel::input_size == kInputFeatures);
static_assert(DirectCBackendModel::output_size == 1U);

template<typename Model>
void run_cpp_direct_c_backend_static_batch(
    const std::array<float, Model::parameter_count>& initial_params,
    std::array<float, Model::parameter_count>& trained_params,
    std::uint32_t seed) {
    static Model model;
    static std::array<float, Model::input_size> input{};
    static std::array<float, Model::output_size> target{};
    static std::array<float, Model::output_size> output_gradient{};

    EDGE_EXPECT_EQ(model.import_parameters(initial_params), edge::Status::Ok);
    EDGE_EXPECT_EQ(model.zero_grad(), edge::Status::Ok);
    EDGE_EXPECT_EQ(model.zero_optimizer_state(), edge::Status::Ok);

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
    std::uint32_t step = 0U;

    for (std::size_t epoch = 0; epoch < kEpochs; ++epoch) {
        for (std::size_t batch = 0; batch < kBatchesPerEpoch; ++batch) {
            EDGE_EXPECT_EQ(model.zero_grad(), edge::Status::Ok);
            const std::size_t begin = batch * kBatchSize;
            for (std::size_t i = 0; i < kBatchSize; ++i) {
                const std::size_t sample = begin + i;
                fill_linear_regression_sample(input, target, sample, seed);
                EDGE_EXPECT_EQ(model.forward(input), edge::Status::Ok);
                edge::TensorView<float, Model::output_size> gradient_view(output_gradient);
                edge::MSE::evaluate(
                    model.output(),
                    edge::TensorView<const float, Model::output_size>(target),
                    gradient_view);
                EDGE_EXPECT_EQ(model.backward(output_gradient), edge::Status::Ok);
            }
            ++step;
            el_backend_adam_update(&params_t, &gradients_t, &m_t, &v_t, kLearningRate, step);
        }
    }
    EDGE_EXPECT_EQ(model.export_parameters(trained_params), edge::Status::Ok);
}

constexpr std::size_t legacy_c_align_up(std::size_t value) noexcept {
    return (value + (EL_ALIGNMENT_BYTES - 1U)) & ~(static_cast<std::size_t>(EL_ALIGNMENT_BYTES) - 1U);
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
    head = legacy_c_tensor_head(head, out_features, in_features); // weights
    head = legacy_c_tensor_head(head, out_features, 1U);          // bias
    head = legacy_c_tensor_head(head, out_features, 1U);          // output
    head = legacy_c_tensor_head(head, out_features, in_features); // grad_weights
    head = legacy_c_tensor_head(head, out_features, 1U);          // grad_bias
    head = legacy_c_tensor_head(head, out_features, in_features); // m_weights
    head = legacy_c_tensor_head(head, out_features, in_features); // v_weights
    head = legacy_c_tensor_head(head, out_features, 1U);          // m_bias
    return legacy_c_tensor_head(head, out_features, 1U);          // v_bias
}

constexpr std::size_t legacy_c_required_arena_bytes() noexcept {
    std::size_t head = 0;
    head = legacy_c_alloc_head(head, 3U * sizeof(el_layer_t));
    head = legacy_c_dense_head(head, kInputFeatures, kHidden1);
    head = legacy_c_dense_head(head, kHidden1, kHidden2);
    head = legacy_c_dense_head(head, kHidden2, 1U);

    std::size_t workspace_cap = kInputFeatures;
    workspace_cap = workspace_cap > kHidden1 ? workspace_cap : kHidden1;
    workspace_cap = workspace_cap > kHidden2 ? workspace_cap : kHidden2;
    head = legacy_c_alloc_head(head, 2U * workspace_cap * sizeof(el_float_t));

    return legacy_c_align_up(head + (EL_ALIGNMENT_BYTES * 4U));
}

#ifdef EDGE_M55_SWEEP_LEGACY_C_ARENA_BYTES
constexpr std::size_t kLegacyCArenaBytes = EDGE_M55_SWEEP_LEGACY_C_ARENA_BYTES;
#else
constexpr std::size_t kLegacyCArenaBytes = legacy_c_required_arena_bytes();
#endif
constexpr std::size_t kLegacyCControlBytes = sizeof(el_context_t) + sizeof(el_network_t);

template<std::size_t ParamCount>
void run_legacy_c_static_batch(const std::array<float, ParamCount>& initial_params,
                               std::array<float, ParamCount>& trained_params,
                               std::uint32_t seed) {
    alignas(EL_ALIGNMENT_BYTES) static std::array<std::uint8_t, kLegacyCArenaBytes> arena{};
    static el_context_t ctx{};
    static el_network_t net{};
    static std::array<float, kInputFeatures> input{};
    static std::array<float, 1> target{};

    arena.fill(0U);
    EDGE_EXPECT_EQ(el_ctx_init(&ctx, arena.data(), static_cast<std::uint32_t>(arena.size())),
                   EL_OK);
    EDGE_EXPECT_EQ(el_network_init(&ctx, &net, 3), EL_OK);
    EDGE_EXPECT_TRUE(el_network_add_dense(
                         &net,
                         static_cast<std::uint16_t>(kInputFeatures),
                         static_cast<std::uint16_t>(kHidden1),
                         EL_ACT_RELU,
                         true,
                         true) != nullptr);
    EDGE_EXPECT_TRUE(el_network_add_dense(
                         &net,
                         static_cast<std::uint16_t>(kHidden1),
                         static_cast<std::uint16_t>(kHidden2),
                         EL_ACT_RELU,
                         true,
                         true) != nullptr);
    EDGE_EXPECT_TRUE(el_network_add_dense(
                         &net,
                         static_cast<std::uint16_t>(kHidden2),
                         1U,
                         EL_ACT_NONE,
                         true,
                         true) != nullptr);
    EDGE_EXPECT_EQ(el_network_prepare_training(&net), EL_OK);
    EDGE_EXPECT_EQ(el_network_param_count(&net), initial_params.size());
    EDGE_EXPECT_TRUE(el_network_import_params(&net, initial_params.data(), initial_params.size()));
    el_network_zero_grad(&net);

    std::uint32_t step = 0U;

    for (std::size_t epoch = 0; epoch < kEpochs; ++epoch) {
        for (std::size_t batch = 0; batch < kBatchesPerEpoch; ++batch) {
            el_network_zero_grad(&net);
            const std::size_t begin = batch * kBatchSize;
            for (std::size_t i = 0; i < kBatchSize; ++i) {
                const std::size_t sample = begin + i;
                fill_linear_regression_sample(input, target, sample, seed);
                el_tensor_t sample_input{
                    .data = input.data(),
                    .rows = static_cast<std::uint16_t>(input.size()),
                    .cols = 1U,
                };
                el_tensor_t sample_target{
                    .data = target.data(),
                    .rows = 1U,
                    .cols = 1U,
                };
                (void)el_network_train_step(&net, &sample_input, &sample_target, EL_LOSS_MSE);
            }
            ++step;
            for (std::uint8_t layer_idx = 0; layer_idx < net.num_layers; ++layer_idx) {
                el_layer_update(&net.layers[layer_idx], kLearningRate, step);
            }
        }
    }

    EDGE_EXPECT_TRUE(el_network_export_params(&net, trained_params.data(), trained_params.size()));
}
#endif

} // namespace

int main() {
    constexpr const char* case_name = "32-" EDGE_M55_SWEEP_STR(EDGE_M55_SWEEP_H1) "x"
                                      EDGE_M55_SWEEP_STR(EDGE_M55_SWEEP_H2) "-1";
    std::array<float, GenericModel::parameter_count> initial_params{};

    for (std::uint32_t seed = 0; seed < kSeedCount; ++seed) {
        fill_initial_params<GenericModel>(initial_params, seed);

#if EDGE_M55_SWEEP_ENABLE_CPP_GENERIC
        std::array<float, GenericModel::parameter_count> generic_params{};
        run_cpp_static_batch<GenericModel>(initial_params, generic_params, seed);
#endif

#if EDGE_M55_SWEEP_ENABLE_CPP_M55
        std::array<float, M55Model::parameter_count> m55_params{};
        run_cpp_static_batch<M55Model>(initial_params, m55_params, seed);
#endif

#if EDGE_M55_SWEEP_ENABLE_CPP_GENERIC && EDGE_M55_SWEEP_ENABLE_CPP_M55
        expect_all_near(case_name,
                        "cpp_m55",
                        m55_params,
                        "cpp_generic",
                        generic_params,
                        kTolerance);
#endif

#if EDGE_M55_SWEEP_ENABLE_LEGACY_C
        std::array<float, GenericModel::parameter_count> legacy_c_params{};
        run_legacy_c_static_batch(initial_params, legacy_c_params, seed);

#if EDGE_M55_SWEEP_ENABLE_DIRECT_C_BACKEND
        std::array<float, DirectCBackendModel::parameter_count> cpp_direct_c_params{};
        run_cpp_direct_c_backend_static_batch<DirectCBackendModel>(
            initial_params, cpp_direct_c_params, seed);
        expect_all_near(case_name,
                        "cpp_direct_c_backend",
                        cpp_direct_c_params,
                        "legacy_c",
                        legacy_c_params,
                        kTolerance);
#endif

#if EDGE_M55_SWEEP_ENABLE_CPP_M55
        expect_all_near(case_name,
                        "legacy_c",
                        legacy_c_params,
                        "cpp_m55",
                        m55_params,
                        kTolerance);
#endif

#if EDGE_M55_SWEEP_ENABLE_CPP_GENERIC
        expect_all_near(case_name,
                        "legacy_c",
                        legacy_c_params,
                        "cpp_generic",
                        generic_params,
                        kTolerance);
#endif
#endif
    }

    std::printf("case=%s seeds=%zu batch=%zu rollout=%zu epochs=%zu "
                "optimizer_steps=%zu sample_passes=%zu params=%zu "
                "legacy_c_arena=%zu legacy_c_control=%zu "
                "cpp_direct_required_memory=%zu cpp_direct_model_object=%zu "
                "cpp_generic_required_memory=%zu cpp_generic_model_object=%zu "
                "cpp_m55_required_memory=%zu cpp_m55_model_object=%zu "
                "m55_mve=%d direct_c=%d legacy_c=%d optimizer=adam opt=Ofast\n",
                case_name,
                kSeedCount,
                kBatchSize,
                kRolloutSamples,
                kEpochs,
                kOptimizerSteps,
                kSamplePasses,
                GenericModel::parameter_count,
 #if EDGE_M55_SWEEP_ENABLE_LEGACY_C
                kLegacyCArenaBytes,
                kLegacyCControlBytes,
                DirectCBackendModel::required_memory,
                sizeof(DirectCBackendModel),
 #else
                std::size_t{0},
                std::size_t{0},
                std::size_t{0},
                std::size_t{0},
 #endif
                GenericModel::required_memory,
                sizeof(GenericModel),
                M55Model::required_memory,
                sizeof(M55Model),
                edge::detail::m55_mve_available ? 1 : 0,
                EDGE_M55_SWEEP_ENABLE_DIRECT_C_BACKEND,
                EDGE_M55_SWEEP_ENABLE_LEGACY_C);
}
