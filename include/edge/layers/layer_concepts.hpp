#pragma once

#include <concepts>
#include <cstddef>

#include <edge/tensor_spec.hpp>

namespace edge {

template<typename T>
concept TensorSpecLike = requires {
    typename T::shape;
    { T::layout } -> std::convertible_to<Layout>;
    { T::rank } -> std::convertible_to<std::size_t>;
    { T::elements } -> std::convertible_to<std::size_t>;
};

template<typename T>
concept InputLayerSpec = requires {
    typename T::spec;
    { T::features } -> std::convertible_to<std::size_t>;
} && TensorSpecLike<typename T::spec>;

template<typename T>
concept DenseLayerSpec = requires {
    { T::out_features } -> std::convertible_to<std::size_t>;
    typename T::activation;
    typename T::initializer;
    typename T::precision_override;
};

template<typename T>
concept LayerInstanceSpec = requires {
    typename T::input_spec;
    typename T::output_spec;
    { T::in_features } -> std::convertible_to<std::size_t>;
    { T::out_features } -> std::convertible_to<std::size_t>;
    { T::parameter_count } -> std::convertible_to<std::size_t>;
    { T::cache_count } -> std::convertible_to<std::size_t>;
    { T::workspace_count } -> std::convertible_to<std::size_t>;
} && TensorSpecLike<typename T::input_spec> && TensorSpecLike<typename T::output_spec>;

template<typename InputSpecT, typename T>
concept CustomLayerSpec = requires {
    typename T::template Instance<InputSpecT>;
} && TensorSpecLike<InputSpecT> && LayerInstanceSpec<typename T::template Instance<InputSpecT>>;

} // namespace edge
