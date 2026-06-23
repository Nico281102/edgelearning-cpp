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

} // namespace edge
