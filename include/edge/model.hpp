#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <edge/activations.hpp>
#include <edge/arena.hpp>
#include <edge/backend.hpp>
#include <edge/config.hpp>
#include <edge/initializers.hpp>
#include <edge/layers/dense.hpp>
#include <edge/layers/layer_concepts.hpp>
#include <edge/memory_plan.hpp>
#include <edge/precision.hpp>
#include <edge/status.hpp>
#include <edge/tensor.hpp>

namespace edge {
namespace detail {

template<typename... Ts>
struct TypeList {};

template<typename T, typename List>
struct TypeListPrepend;

template<typename T, typename... Ts>
struct TypeListPrepend<T, TypeList<Ts...>> {
    using type = TypeList<T, Ts...>;
};

template<typename... Specs>
struct ModelArgs;

template<typename InputLayer, typename... Layers>
    requires InputSpec<InputLayer>
struct ModelArgs<InputLayer, Layers...> {
    using backend = Backend::Default;
    using input = InputLayer;
    using layers = TypeList<Layers...>;
};

template<typename BackendPolicyT, typename InputLayer, typename... Layers>
    requires BackendPolicy<BackendPolicyT> && InputSpec<InputLayer>
struct ModelArgs<BackendPolicyT, InputLayer, Layers...> {
    using backend = BackendPolicyT;
    using input = InputLayer;
    using layers = TypeList<Layers...>;
};

template<std::size_t CurrentFeatures, typename... Layers>
struct LayerChain;

template<std::size_t CurrentFeatures>
struct LayerChain<CurrentFeatures> {
    using instances = TypeList<>;
    static constexpr std::size_t layer_count = 0;
    static constexpr std::size_t output_features = CurrentFeatures;
    static constexpr std::size_t parameter_count = 0;
    static constexpr std::size_t output_activation_count = 0;
    static constexpr std::size_t preactivation_count = 0;
    static constexpr std::size_t max_features = CurrentFeatures;
};

template<std::size_t CurrentFeatures, typename Layer, typename... Rest>
    requires DenseLayerSpec<Layer>
struct LayerChain<CurrentFeatures, Layer, Rest...> {
    using instance = DenseInstance<CurrentFeatures, Layer>;
    using tail = LayerChain<Layer::out_features, Rest...>;
    using instances = typename TypeListPrepend<instance, typename tail::instances>::type;

    static constexpr std::size_t layer_count = 1U + tail::layer_count;
    static constexpr std::size_t output_features = tail::output_features;
    static constexpr std::size_t parameter_count =
        instance::parameter_count + tail::parameter_count;
    static constexpr std::size_t output_activation_count =
        instance::out_features + tail::output_activation_count;
    static constexpr std::size_t preactivation_count =
        instance::preactivation_count + tail::preactivation_count;
    static constexpr std::size_t local_max =
        CurrentFeatures > instance::out_features ? CurrentFeatures : instance::out_features;
    static constexpr std::size_t max_features =
        local_max > tail::max_features ? local_max : tail::max_features;
};

template<typename BackendPolicyT, typename InputLayer, typename LayerList>
class ModelImpl;

template<typename BackendPolicyT, typename InputLayer, typename... Layers>
class ModelImpl<BackendPolicyT, InputLayer, TypeList<Layers...>> {
    static_assert(sizeof...(Layers) > 0, "Model requires at least one trainable layer");
    static_assert((DenseLayerSpec<Layers> && ...),
                  "EdgeLearning++ v0.1 supports Dense layers only");

    using Chain = LayerChain<InputLayer::features, Layers...>;
    using Instances = typename Chain::instances;

public:
    using backend = BackendPolicyT;
    using precision = edge::precision::FP32;
    using scalar_type = float;

    static constexpr std::size_t input_size = InputLayer::features;
    static constexpr std::size_t output_size = Chain::output_features;
    static constexpr std::size_t layer_count = Chain::layer_count;
    static constexpr std::size_t parameter_count = Chain::parameter_count;
    static constexpr std::size_t gradient_count = parameter_count;
    static constexpr std::size_t optimizer_state_count = parameter_count * 2U;
    static constexpr std::size_t output_activation_count =
        input_size + Chain::output_activation_count;
    static constexpr std::size_t preactivation_count = Chain::preactivation_count;
    static constexpr std::size_t activation_count =
        output_activation_count + preactivation_count;
    static constexpr std::size_t workspace_count = Chain::max_features * 2U;
    static constexpr std::size_t max_features = Chain::max_features;
    static constexpr std::size_t alignment = alignof(float);

    static constexpr std::size_t parameter_bytes = parameter_count * sizeof(float);
    static constexpr std::size_t gradient_bytes = gradient_count * sizeof(float);
    static constexpr std::size_t optimizer_bytes = optimizer_state_count * sizeof(float);
    static constexpr std::size_t activation_bytes = activation_count * sizeof(float);
    static constexpr std::size_t workspace_bytes = workspace_count * sizeof(float);
    static constexpr std::size_t total_bytes =
        parameter_bytes + gradient_bytes + optimizer_bytes + activation_bytes + workspace_bytes;
    static constexpr std::size_t required_memory = align_up(total_bytes, alignment);

    ModelImpl() noexcept {
        bind_arena(internal_arena_.data(), internal_arena_.size());
        clear_all();
    }

    template<std::size_t N>
    explicit ModelImpl(ExternalArena<N> arena) noexcept {
        static_assert(N >= required_memory,
                      "External arena is too small for this model; use Model::required_memory");
        bind_arena(arena.data, N);
        clear_all();
    }

    ModelImpl(const ModelImpl&) = delete;
    ModelImpl& operator=(const ModelImpl&) = delete;
    ModelImpl(ModelImpl&&) = delete;
    ModelImpl& operator=(ModelImpl&&) = delete;

    Status status() const noexcept {
        return status_;
    }

    static constexpr MemoryBreakdown memory_breakdown() noexcept {
        return MemoryBreakdown{
            parameter_bytes,
            gradient_bytes,
            optimizer_bytes,
            activation_bytes,
            workspace_bytes,
            total_bytes};
    }

    Status initialize(const InitConfig& config = {}) noexcept {
        if (status_ != Status::Ok) {
            return status_;
        }
        DeterministicRng rng(config.seed);
        initialize_layers<0>(rng, config, Instances{});
        zero_grad();
        zero_optimizer_state();
        return Status::Ok;
    }

    Status zero_grad() noexcept {
        if (gradients_ == nullptr) {
            return Status::NullPointer;
        }
        for (std::size_t i = 0; i < gradient_count; ++i) {
            gradients_[i] = 0.0F;
        }
        return Status::Ok;
    }

    Status zero_optimizer_state() noexcept {
        if (optimizer_state_ == nullptr) {
            return Status::NullPointer;
        }
        for (std::size_t i = 0; i < optimizer_state_count; ++i) {
            optimizer_state_[i] = 0.0F;
        }
        return Status::Ok;
    }

    Status forward(TensorView<const float, input_size> input) noexcept {
        if (status_ != Status::Ok) {
            return status_;
        }
        if (!input) {
            return Status::NullPointer;
        }
        for (std::size_t i = 0; i < input_size; ++i) {
            activations_[i] = input[i];
        }
        forward_entry(Instances{});
        return Status::Ok;
    }

    Status forward(const std::array<float, input_size>& input) noexcept {
        return forward(TensorView<const float, input_size>(input));
    }

    Status forward(const float (&input)[input_size]) noexcept {
        return forward(TensorView<const float, input_size>(input));
    }

    Status backward(TensorView<const float, output_size> output_gradient) noexcept {
        if (status_ != Status::Ok) {
            return status_;
        }
        if (!output_gradient) {
            return Status::NullPointer;
        }

        float* upstream = workspace_;
        float* downstream = workspace_ + max_features;
        for (std::size_t i = 0; i < output_size; ++i) {
            upstream[i] = output_gradient[i];
        }
        backward_entry(upstream, downstream, Instances{});
        return Status::Ok;
    }

    Status backward(const std::array<float, output_size>& output_gradient) noexcept {
        return backward(TensorView<const float, output_size>(output_gradient));
    }

    Status backward(const float (&output_gradient)[output_size]) noexcept {
        return backward(TensorView<const float, output_size>(output_gradient));
    }

    TensorView<float, output_size> output() noexcept {
        return TensorView<float, output_size>(activations_ + output_offset());
    }

    TensorView<const float, output_size> output() const noexcept {
        return TensorView<const float, output_size>(activations_ + output_offset());
    }

    float* parameter_data() noexcept {
        return parameters_;
    }

    const float* parameter_data() const noexcept {
        return parameters_;
    }

    float* gradient_data() noexcept {
        return gradients_;
    }

    const float* gradient_data() const noexcept {
        return gradients_;
    }

    float* optimizer_state_data() noexcept {
        return optimizer_state_;
    }

    const float* optimizer_state_data() const noexcept {
        return optimizer_state_;
    }

    template<std::size_t N>
    Status export_parameters(std::array<float, N>& output) const noexcept {
        static_assert(N >= parameter_count,
                      "Export buffer must hold at least Model::parameter_count floats");
        return export_parameters(output.data(), N);
    }

    template<std::size_t N>
    Status export_parameters(float (&output)[N]) const noexcept {
        static_assert(N >= parameter_count,
                      "Export buffer must hold at least Model::parameter_count floats");
        return export_parameters(output, N);
    }

    Status export_parameters(float* output, std::size_t count) const noexcept {
        if (output == nullptr) {
            return Status::NullPointer;
        }
        if (count < parameter_count) {
            return Status::InvalidBufferLength;
        }
        for (std::size_t i = 0; i < parameter_count; ++i) {
            output[i] = parameters_[i];
        }
        return Status::Ok;
    }

    template<std::size_t N>
    Status import_parameters(const std::array<float, N>& input) noexcept {
        static_assert(N >= parameter_count,
                      "Import buffer must hold at least Model::parameter_count floats");
        return import_parameters(input.data(), N);
    }

    template<std::size_t N>
    Status import_parameters(const float (&input)[N]) noexcept {
        static_assert(N >= parameter_count,
                      "Import buffer must hold at least Model::parameter_count floats");
        return import_parameters(input, N);
    }

    Status import_parameters(const float* input, std::size_t count) noexcept {
        if (input == nullptr) {
            return Status::NullPointer;
        }
        if (count < parameter_count) {
            return Status::InvalidBufferLength;
        }
        for (std::size_t i = 0; i < parameter_count; ++i) {
            parameters_[i] = input[i];
        }
        return Status::Ok;
    }

private:
    static constexpr std::size_t output_offset() noexcept {
        return output_activation_count - output_size;
    }

    float* preactivation_base() noexcept {
        return activations_ + output_activation_count;
    }

    const float* preactivation_base() const noexcept {
        return activations_ + output_activation_count;
    }

    void bind_arena(std::byte* arena, std::size_t bytes) noexcept {
        if (arena == nullptr) {
            status_ = Status::NullPointer;
            return;
        }
        if (bytes < required_memory) {
            status_ = Status::InsufficientArena;
            return;
        }
        const auto address = reinterpret_cast<std::uintptr_t>(arena);
        if ((address % alignment) != 0U) {
            status_ = Status::UnalignedArena;
            return;
        }

        float* base = reinterpret_cast<float*>(arena);
        parameters_ = base;
        gradients_ = parameters_ + parameter_count;
        optimizer_state_ = gradients_ + gradient_count;
        activations_ = optimizer_state_ + optimizer_state_count;
        workspace_ = activations_ + activation_count;
        status_ = Status::Ok;
    }

    void clear_all() noexcept {
        if (status_ != Status::Ok) {
            return;
        }
        float* base = parameters_;
        const std::size_t count =
            parameter_count + gradient_count + optimizer_state_count + activation_count + workspace_count;
        for (std::size_t i = 0; i < count; ++i) {
            base[i] = 0.0F;
        }
    }

    template<std::size_t ParamOffset, typename Instance>
    void initialize_one_layer(DeterministicRng& rng, const InitConfig& config) noexcept {
        using Initializer = typename Instance::initializer;
        float* weights = parameters_ + ParamOffset;
        float* bias = weights + Instance::weight_count;
        Initializer::template fill<Instance::in_features, Instance::out_features>(
            weights, rng, config);
        for (std::size_t i = 0; i < Instance::bias_count; ++i) {
            bias[i] = config.bias;
        }
    }

    template<std::size_t ParamOffset>
    void initialize_layers(DeterministicRng&, const InitConfig&, TypeList<>) noexcept {}

    template<std::size_t ParamOffset, typename Instance, typename... Tail>
    void initialize_layers(DeterministicRng& rng,
                           const InitConfig& config,
                           TypeList<Instance, Tail...>) noexcept {
        initialize_one_layer<ParamOffset, Instance>(rng, config);
        initialize_layers<ParamOffset + Instance::parameter_count>(rng, config, TypeList<Tail...>{});
    }

    template<std::size_t ParamOffset, typename List>
    void initialize_layers(DeterministicRng& rng, const InitConfig& config) noexcept {
        initialize_layers<ParamOffset>(rng, config, List{});
    }

    template<typename... InstancesT>
    void forward_entry(TypeList<InstancesT...>) noexcept {
        forward_layers<0, input_size, 0, InstancesT...>(activations_);
    }

    template<std::size_t, std::size_t, std::size_t>
    void forward_layers(const float*) noexcept {}

    template<
        std::size_t ParamOffset,
        std::size_t ActOutOffset,
        std::size_t PreOffset,
        typename Instance,
        typename... Tail>
    void forward_layers(const float* previous) noexcept {
        using Activation = typename Instance::activation;
        const float* weights = parameters_ + ParamOffset;
        const float* bias = weights + Instance::weight_count;
        float* output_values = activations_ + ActOutOffset;
        float* pre_values = Instance::stores_preactivation ? preactivation_base() + PreOffset : nullptr;

        for (std::size_t out = 0; out < Instance::out_features; ++out) {
            float z = bias[out];
            const std::size_t row = out * Instance::in_features;
            for (std::size_t in = 0; in < Instance::in_features; ++in) {
                z += weights[row + in] * previous[in];
            }
            if constexpr (Instance::stores_preactivation) {
                pre_values[out] = z;
            }
            output_values[out] = Activation::template forward<float>(z);
        }

        if constexpr (sizeof...(Tail) > 0) {
            forward_layers<
                ParamOffset + Instance::parameter_count,
                ActOutOffset + Instance::out_features,
                PreOffset + Instance::preactivation_count,
                Tail...>(output_values);
        }
    }

    template<typename... InstancesT>
    void backward_entry(float*& upstream, float*& downstream, TypeList<InstancesT...>) noexcept {
        backward_layers<0, 0, input_size, 0, 0, InstancesT...>(upstream, downstream);
    }

    template<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>
    void backward_layers(float*&, float*&) noexcept {}

    template<
        std::size_t ParamOffset,
        std::size_t GradOffset,
        std::size_t ActOutOffset,
        std::size_t PrevActOffset,
        std::size_t PreOffset,
        typename Instance,
        typename... Tail>
    void backward_layers(float*& upstream, float*& downstream) noexcept {
        if constexpr (sizeof...(Tail) > 0) {
            backward_layers<
                ParamOffset + Instance::parameter_count,
                GradOffset + Instance::parameter_count,
                ActOutOffset + Instance::out_features,
                ActOutOffset,
                PreOffset + Instance::preactivation_count,
                Tail...>(upstream, downstream);
        }

        using Activation = typename Instance::activation;
        const float* previous_activation = activations_ + PrevActOffset;
        const float* layer_output = activations_ + ActOutOffset;
        const float* pre_values =
            Instance::stores_preactivation ? preactivation_base() + PreOffset : nullptr;
        const float* weights = parameters_ + ParamOffset;
        float* grad_weights = gradients_ + GradOffset;
        float* grad_bias = grad_weights + Instance::weight_count;

        for (std::size_t i = 0; i < Instance::in_features; ++i) {
            downstream[i] = 0.0F;
        }

        for (std::size_t out = 0; out < Instance::out_features; ++out) {
            const float z = Instance::stores_preactivation ? pre_values[out] : 0.0F;
            const float deriv = activation_derivative<Activation, float>(z, layer_output[out]);
            const float delta = upstream[out] * deriv;
            grad_bias[out] += delta;
            const std::size_t row = out * Instance::in_features;
            for (std::size_t in = 0; in < Instance::in_features; ++in) {
                grad_weights[row + in] += delta * previous_activation[in];
                downstream[in] += weights[row + in] * delta;
            }
        }

        float* old_upstream = upstream;
        upstream = downstream;
        downstream = old_upstream;
    }

    alignas(alignment) std::array<std::byte, required_memory> internal_arena_{};
    Status status_ = Status::NotInitialized;
    float* parameters_ = nullptr;
    float* gradients_ = nullptr;
    float* optimizer_state_ = nullptr;
    float* activations_ = nullptr;
    float* workspace_ = nullptr;
};

} // namespace detail

template<typename... Specs>
class Model
    : public detail::ModelImpl<
          typename detail::ModelArgs<Specs...>::backend,
          typename detail::ModelArgs<Specs...>::input,
          typename detail::ModelArgs<Specs...>::layers> {
    using Base = detail::ModelImpl<
        typename detail::ModelArgs<Specs...>::backend,
        typename detail::ModelArgs<Specs...>::input,
        typename detail::ModelArgs<Specs...>::layers>;

public:
    using Base::Base;
};

} // namespace edge
