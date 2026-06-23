#pragma once

#include <cstddef>

#include <edge/activations.hpp>
#include <edge/initializers.hpp>
#include <edge/precision.hpp>

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
};

} // namespace detail

} // namespace edge

