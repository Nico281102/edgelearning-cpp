#pragma once

#include <concepts>
#include <cstddef>

namespace edge {

template<typename T>
concept InputSpec = requires {
    { T::features } -> std::convertible_to<std::size_t>;
};

template<typename T>
concept DenseLayerSpec = requires {
    { T::out_features } -> std::convertible_to<std::size_t>;
    typename T::activation;
    typename T::initializer;
    typename T::precision_override;
};

template<typename T>
concept LayerInstanceSpec = requires {
    { T::in_features } -> std::convertible_to<std::size_t>;
    { T::out_features } -> std::convertible_to<std::size_t>;
    { T::parameter_count } -> std::convertible_to<std::size_t>;
    { T::cache_count } -> std::convertible_to<std::size_t>;
    { T::workspace_count } -> std::convertible_to<std::size_t>;
};

template<std::size_t InFeatures, typename T>
concept CustomLayerSpec = requires {
    typename T::template Instance<InFeatures>;
} && LayerInstanceSpec<typename T::template Instance<InFeatures>>;

} // namespace edge
