#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

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

template<std::size_t InFeatures, typename LayerSpec, typename = void>
struct MakeLayerInstance {
    static_assert(sizeof(LayerSpec) == 0,
                  "Layer must be edge::Dense or provide template<std::size_t In> struct Instance");
};

template<std::size_t InFeatures, typename DenseSpec>
    requires DenseLayerSpec<DenseSpec>
struct MakeLayerInstance<InFeatures, DenseSpec> {
    using type = DenseInstance<InFeatures, DenseSpec>;
};

template<std::size_t InFeatures, typename LayerSpec>
    requires CustomLayerSpec<InFeatures, LayerSpec> && (!DenseLayerSpec<LayerSpec>)
struct MakeLayerInstance<InFeatures, LayerSpec> {
    using type = typename LayerSpec::template Instance<InFeatures>;
};

template<typename... Specs>
struct ModelArgs;

template<typename InputLayer, typename... Layers>
    requires InputSpec<InputLayer>
struct ModelArgs<InputLayer, Layers...> {
    using backend = Backend::Default;
    using precision = edge::precision::FP32;
    using input = InputLayer;
    using layers = TypeList<Layers...>;
};

template<typename PrecisionT, typename InputLayer, typename... Layers>
    requires PrecisionPolicy<PrecisionT> && InputSpec<InputLayer>
struct ModelArgs<PrecisionT, InputLayer, Layers...> {
    using backend = Backend::Default;
    using precision = PrecisionT;
    using input = InputLayer;
    using layers = TypeList<Layers...>;
};

template<typename BackendPolicyT, typename InputLayer, typename... Layers>
    requires BackendPolicy<BackendPolicyT> && InputSpec<InputLayer>
struct ModelArgs<BackendPolicyT, InputLayer, Layers...> {
    using backend = BackendPolicyT;
    using precision = edge::precision::FP32;
    using input = InputLayer;
    using layers = TypeList<Layers...>;
};

template<typename BackendPolicyT, typename PrecisionT, typename InputLayer, typename... Layers>
    requires BackendPolicy<BackendPolicyT> && PrecisionPolicy<PrecisionT> && InputSpec<InputLayer>
struct ModelArgs<BackendPolicyT, PrecisionT, InputLayer, Layers...> {
    using backend = BackendPolicyT;
    using precision = PrecisionT;
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
    static constexpr std::size_t cache_count = 0;
    static constexpr std::size_t layer_workspace_count = 0;
    static constexpr std::size_t max_features = CurrentFeatures;
};

template<std::size_t CurrentFeatures, typename Layer, typename... Rest>
struct LayerChain<CurrentFeatures, Layer, Rest...> {
    using instance = typename MakeLayerInstance<CurrentFeatures, Layer>::type;
    static_assert(LayerInstanceSpec<instance>,
                  "Layer instance must expose shape, parameter, cache, and workspace constants");

    using tail = LayerChain<instance::out_features, Rest...>;
    using instances = typename TypeListPrepend<instance, typename tail::instances>::type;

    static constexpr std::size_t layer_count = 1U + tail::layer_count;
    static constexpr std::size_t output_features = tail::output_features;
    static constexpr std::size_t parameter_count =
        instance::parameter_count + tail::parameter_count;
    static constexpr std::size_t output_activation_count =
        instance::out_features + tail::output_activation_count;
    static constexpr std::size_t cache_count = instance::cache_count + tail::cache_count;
    static constexpr std::size_t layer_workspace_count =
        instance::workspace_count > tail::layer_workspace_count
            ? instance::workspace_count
            : tail::layer_workspace_count;
    static constexpr std::size_t local_max =
        CurrentFeatures > instance::out_features ? CurrentFeatures : instance::out_features;
    static constexpr std::size_t max_features =
        local_max > tail::max_features ? local_max : tail::max_features;
};

template<typename BackendPolicyT, typename PrecisionT>
struct ModelTypes {
    using BackendT = BackendPolicyT;
    using ParameterT = typename PrecisionT::ParameterT;
    using ActivationT = typename PrecisionT::ActivationT;
    using GradientT = typename PrecisionT::GradientT;
    using AccumulatorT = typename PrecisionT::AccumulatorT;
    using OptimizerStateT = typename PrecisionT::OptimizerStateT;
    using LossT = typename PrecisionT::LossT;
};

template<typename BackendPolicyT, typename PrecisionT, typename InputLayer, typename LayerList>
class ModelImpl;

template<typename BackendPolicyT, typename PrecisionT, typename InputLayer, typename... Layers>
class ModelImpl<BackendPolicyT, PrecisionT, InputLayer, TypeList<Layers...>> {
    static_assert(sizeof...(Layers) > 0, "Model requires at least one trainable layer");

    using Chain = LayerChain<InputLayer::features, Layers...>;
    using Instances = typename Chain::instances;

public:
    using backend = BackendPolicyT;
    using precision = PrecisionT;
    using types = ModelTypes<BackendPolicyT, PrecisionT>;
    using parameter_type = typename types::ParameterT;
    using activation_type = typename types::ActivationT;
    using gradient_type = typename types::GradientT;
    using accumulator_type = typename types::AccumulatorT;
    using optimizer_state_type = typename types::OptimizerStateT;
    using loss_type = typename types::LossT;
    using scalar_type = activation_type;

    static constexpr std::size_t input_size = InputLayer::features;
    static constexpr std::size_t output_size = Chain::output_features;
    static constexpr std::size_t layer_count = Chain::layer_count;
    static constexpr std::size_t parameter_count = Chain::parameter_count;
    static constexpr std::size_t gradient_count = parameter_count;
    static constexpr std::size_t optimizer_state_count = parameter_count * 2U;
    static constexpr std::size_t output_activation_count =
        input_size + Chain::output_activation_count;
    static constexpr std::size_t cache_count = Chain::cache_count;
    static constexpr std::size_t preactivation_count = cache_count;
    static constexpr std::size_t activation_count = output_activation_count + cache_count;
    static constexpr std::size_t workspace_count = Chain::max_features * 2U + Chain::layer_workspace_count;
    static constexpr std::size_t max_features = Chain::max_features;
    static constexpr std::size_t layer_workspace_offset = Chain::max_features * 2U;
    static constexpr std::size_t backend_alignment =
        std::is_same_v<BackendPolicyT, Backend::M55> ? 32U : 1U;
    static constexpr std::size_t alignment = static_max_v<
        backend_alignment,
        alignof(parameter_type),
        alignof(gradient_type),
        alignof(optimizer_state_type),
        alignof(activation_type),
        alignof(accumulator_type)>;

    static constexpr std::size_t parameter_bytes = parameter_count * sizeof(parameter_type);
    static constexpr std::size_t gradient_bytes = gradient_count * sizeof(gradient_type);
    static constexpr std::size_t optimizer_bytes =
        optimizer_state_count * sizeof(optimizer_state_type);
    static constexpr std::size_t activation_bytes = activation_count * sizeof(activation_type);
    static constexpr std::size_t workspace_bytes = workspace_count * sizeof(accumulator_type);

    static constexpr std::size_t parameter_offset = 0;
    static constexpr std::size_t gradient_offset =
        align_up(parameter_offset + parameter_bytes, alignment);
    static constexpr std::size_t optimizer_offset =
        align_up(gradient_offset + gradient_bytes, alignment);
    static constexpr std::size_t activation_offset =
        align_up(optimizer_offset + optimizer_bytes, alignment);
    static constexpr std::size_t workspace_offset =
        align_up(activation_offset + activation_bytes, alignment);
    static constexpr std::size_t total_bytes = workspace_offset + workspace_bytes;
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
            gradients_[i] = gradient_type{0};
        }
        return Status::Ok;
    }

    Status zero_optimizer_state() noexcept {
        if (optimizer_state_ == nullptr) {
            return Status::NullPointer;
        }
        for (std::size_t i = 0; i < optimizer_state_count; ++i) {
            optimizer_state_[i] = optimizer_state_type{0};
        }
        return Status::Ok;
    }

    Status forward(TensorView<const activation_type, input_size> input) noexcept {
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

    Status forward(const std::array<activation_type, input_size>& input) noexcept {
        return forward(TensorView<const activation_type, input_size>(input));
    }

    Status forward(const activation_type (&input)[input_size]) noexcept {
        return forward(TensorView<const activation_type, input_size>(input));
    }

    Status backward(TensorView<const accumulator_type, output_size> output_gradient) noexcept {
        if (status_ != Status::Ok) {
            return status_;
        }
        if (!output_gradient) {
            return Status::NullPointer;
        }

        accumulator_type* upstream = workspace_;
        accumulator_type* downstream = workspace_ + max_features;
        for (std::size_t i = 0; i < output_size; ++i) {
            upstream[i] = output_gradient[i];
        }
        backward_entry(upstream, downstream, Instances{});
        return Status::Ok;
    }

    Status backward(const std::array<accumulator_type, output_size>& output_gradient) noexcept {
        return backward(TensorView<const accumulator_type, output_size>(output_gradient));
    }

    Status backward(const accumulator_type (&output_gradient)[output_size]) noexcept {
        return backward(TensorView<const accumulator_type, output_size>(output_gradient));
    }

    TensorView<activation_type, output_size> output() noexcept {
        return TensorView<activation_type, output_size>(activations_ + output_offset());
    }

    TensorView<const activation_type, output_size> output() const noexcept {
        return TensorView<const activation_type, output_size>(activations_ + output_offset());
    }

    parameter_type* parameter_data() noexcept {
        return parameters_;
    }

    const parameter_type* parameter_data() const noexcept {
        return parameters_;
    }

    gradient_type* gradient_data() noexcept {
        return gradients_;
    }

    const gradient_type* gradient_data() const noexcept {
        return gradients_;
    }

    optimizer_state_type* optimizer_state_data() noexcept {
        return optimizer_state_;
    }

    const optimizer_state_type* optimizer_state_data() const noexcept {
        return optimizer_state_;
    }

    template<std::size_t N>
    Status export_parameters(std::array<parameter_type, N>& output) const noexcept {
        static_assert(N >= parameter_count,
                      "Export buffer must hold at least Model::parameter_count elements");
        return export_parameters(output.data(), N);
    }

    template<std::size_t N>
    Status export_parameters(parameter_type (&output)[N]) const noexcept {
        static_assert(N >= parameter_count,
                      "Export buffer must hold at least Model::parameter_count elements");
        return export_parameters(output, N);
    }

    Status export_parameters(parameter_type* output, std::size_t count) const noexcept {
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
    Status import_parameters(const std::array<parameter_type, N>& input) noexcept {
        static_assert(N >= parameter_count,
                      "Import buffer must hold at least Model::parameter_count elements");
        return import_parameters(input.data(), N);
    }

    template<std::size_t N>
    Status import_parameters(const parameter_type (&input)[N]) noexcept {
        static_assert(N >= parameter_count,
                      "Import buffer must hold at least Model::parameter_count elements");
        return import_parameters(input, N);
    }

    Status import_parameters(const parameter_type* input, std::size_t count) noexcept {
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

    activation_type* cache_base() noexcept {
        return activations_ + output_activation_count;
    }

    const activation_type* cache_base() const noexcept {
        return activations_ + output_activation_count;
    }

    accumulator_type* layer_workspace_base() noexcept {
        return workspace_ + layer_workspace_offset;
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

        parameters_ = reinterpret_cast<parameter_type*>(arena + parameter_offset);
        gradients_ = reinterpret_cast<gradient_type*>(arena + gradient_offset);
        optimizer_state_ = reinterpret_cast<optimizer_state_type*>(arena + optimizer_offset);
        activations_ = reinterpret_cast<activation_type*>(arena + activation_offset);
        workspace_ = reinterpret_cast<accumulator_type*>(arena + workspace_offset);
        status_ = Status::Ok;
    }

    void clear_all() noexcept {
        if (status_ != Status::Ok) {
            return;
        }
        for (std::size_t i = 0; i < parameter_count; ++i) {
            parameters_[i] = parameter_type{0};
        }
        for (std::size_t i = 0; i < gradient_count; ++i) {
            gradients_[i] = gradient_type{0};
        }
        for (std::size_t i = 0; i < optimizer_state_count; ++i) {
            optimizer_state_[i] = optimizer_state_type{0};
        }
        for (std::size_t i = 0; i < activation_count; ++i) {
            activations_[i] = activation_type{0};
        }
        for (std::size_t i = 0; i < workspace_count; ++i) {
            workspace_[i] = accumulator_type{0};
        }
    }

    template<std::size_t ParamOffset, typename Instance>
    void initialize_one_layer(DeterministicRng& rng, const InitConfig& config) noexcept {
        Instance::template initialize<types>(
            TensorView<parameter_type, Instance::parameter_count>(parameters_ + ParamOffset),
            rng,
            config);
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

    template<typename... InstancesT>
    void forward_entry(TypeList<InstancesT...>) noexcept {
        forward_layers<0, input_size, 0, 0, InstancesT...>(activations_);
    }

    template<std::size_t, std::size_t, std::size_t, std::size_t>
    void forward_layers(const activation_type*) noexcept {}

    template<
        std::size_t ParamOffset,
        std::size_t ActOutOffset,
        std::size_t CacheOffset,
        std::size_t WorkspaceOffset,
        typename Instance,
        typename... Tail>
    void forward_layers(const activation_type* previous) noexcept {
        activation_type* output_values = activations_ + ActOutOffset;
        Instance::template forward<types>(
            TensorView<const activation_type, Instance::in_features>(previous),
            TensorView<activation_type, Instance::out_features>(output_values),
            TensorView<const parameter_type, Instance::parameter_count>(parameters_ + ParamOffset),
            TensorView<activation_type, Instance::cache_count>(cache_base() + CacheOffset),
            TensorView<accumulator_type, Instance::workspace_count>(
                layer_workspace_base() + WorkspaceOffset));

        if constexpr (sizeof...(Tail) > 0) {
            forward_layers<
                ParamOffset + Instance::parameter_count,
                ActOutOffset + Instance::out_features,
                CacheOffset + Instance::cache_count,
                WorkspaceOffset,
                Tail...>(output_values);
        }
    }

    template<typename... InstancesT>
    void backward_entry(accumulator_type*& upstream,
                        accumulator_type*& downstream,
                        TypeList<InstancesT...>) noexcept {
        backward_layers<0, 0, input_size, 0, 0, 0, InstancesT...>(upstream, downstream);
    }

    template<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>
    void backward_layers(accumulator_type*&, accumulator_type*&) noexcept {}

    template<
        std::size_t ParamOffset,
        std::size_t GradOffset,
        std::size_t ActOutOffset,
        std::size_t PrevActOffset,
        std::size_t CacheOffset,
        std::size_t WorkspaceOffset,
        typename Instance,
        typename... Tail>
    void backward_layers(accumulator_type*& upstream, accumulator_type*& downstream) noexcept {
        if constexpr (sizeof...(Tail) > 0) {
            backward_layers<
                ParamOffset + Instance::parameter_count,
                GradOffset + Instance::parameter_count,
                ActOutOffset + Instance::out_features,
                ActOutOffset,
                CacheOffset + Instance::cache_count,
                WorkspaceOffset,
                Tail...>(upstream, downstream);
        }

        backward_one_layer<
            ParamOffset,
            GradOffset,
            ActOutOffset,
            PrevActOffset,
            CacheOffset,
            WorkspaceOffset,
            Instance>(upstream, downstream);
    }

    template<
        std::size_t ParamOffset,
        std::size_t GradOffset,
        std::size_t ActOutOffset,
        std::size_t PrevActOffset,
        std::size_t CacheOffset,
        std::size_t WorkspaceOffset,
        typename Instance>
    void backward_one_layer(accumulator_type*& upstream, accumulator_type*& downstream) noexcept {
        if constexpr (ParamOffset == 0U && PrevActOffset == 0U && requires {
                          Instance::template backward_inputless<types>(
                              TensorView<const activation_type, Instance::in_features>(
                                  activations_ + PrevActOffset),
                              TensorView<const activation_type, Instance::out_features>(
                                  activations_ + ActOutOffset),
                              TensorView<const accumulator_type, Instance::out_features>(upstream),
                              TensorView<const parameter_type, Instance::parameter_count>(
                                  parameters_ + ParamOffset),
                              TensorView<gradient_type, Instance::parameter_count>(
                                  gradients_ + GradOffset),
                              TensorView<const activation_type, Instance::cache_count>(
                                  cache_base() + CacheOffset),
                              TensorView<accumulator_type, Instance::workspace_count>(
                                  layer_workspace_base() + WorkspaceOffset));
                      }) {
            Instance::template backward_inputless<types>(
                TensorView<const activation_type, Instance::in_features>(
                    activations_ + PrevActOffset),
                TensorView<const activation_type, Instance::out_features>(
                    activations_ + ActOutOffset),
                TensorView<const accumulator_type, Instance::out_features>(upstream),
                TensorView<const parameter_type, Instance::parameter_count>(
                    parameters_ + ParamOffset),
                TensorView<gradient_type, Instance::parameter_count>(gradients_ + GradOffset),
                TensorView<const activation_type, Instance::cache_count>(
                    cache_base() + CacheOffset),
                TensorView<accumulator_type, Instance::workspace_count>(
                    layer_workspace_base() + WorkspaceOffset));
        } else {
            Instance::template backward<types>(
                TensorView<const activation_type, Instance::in_features>(
                    activations_ + PrevActOffset),
                TensorView<const activation_type, Instance::out_features>(
                    activations_ + ActOutOffset),
                TensorView<const accumulator_type, Instance::out_features>(upstream),
                TensorView<accumulator_type, Instance::in_features>(downstream),
                TensorView<const parameter_type, Instance::parameter_count>(
                    parameters_ + ParamOffset),
                TensorView<gradient_type, Instance::parameter_count>(gradients_ + GradOffset),
                TensorView<const activation_type, Instance::cache_count>(
                    cache_base() + CacheOffset),
                TensorView<accumulator_type, Instance::workspace_count>(
                    layer_workspace_base() + WorkspaceOffset));
        }

        accumulator_type* old_upstream = upstream;
        upstream = downstream;
        downstream = old_upstream;
    }

    alignas(alignment) std::array<std::byte, required_memory> internal_arena_{};
    Status status_ = Status::NotInitialized;
    parameter_type* parameters_ = nullptr;
    gradient_type* gradients_ = nullptr;
    optimizer_state_type* optimizer_state_ = nullptr;
    activation_type* activations_ = nullptr;
    accumulator_type* workspace_ = nullptr;
};

} // namespace detail

template<typename... Specs>
class Model
    : public detail::ModelImpl<
          typename detail::ModelArgs<Specs...>::backend,
          typename detail::ModelArgs<Specs...>::precision,
          typename detail::ModelArgs<Specs...>::input,
          typename detail::ModelArgs<Specs...>::layers> {
    using Base = detail::ModelImpl<
        typename detail::ModelArgs<Specs...>::backend,
        typename detail::ModelArgs<Specs...>::precision,
        typename detail::ModelArgs<Specs...>::input,
        typename detail::ModelArgs<Specs...>::layers>;

public:
    using Base::Base;
};

} // namespace edge
