#pragma once

#include <concepts>
#include <type_traits>

namespace edge {

struct UseModelPrecision {};

namespace precision {

struct FP32 {
    using ParameterT = float;
    using ActivationT = float;
    using GradientT = float;
    using AccumulatorT = float;
    using OptimizerStateT = float;
    using LossT = float;
};

struct MixedFP16 {
    using ParameterT = float;
    using ActivationT = float;
    using GradientT = float;
    using AccumulatorT = float;
    using OptimizerStateT = float;
    using LossT = float;
};

} // namespace precision

template<typename T>
concept PrecisionPolicy = requires {
    typename T::ParameterT;
    typename T::ActivationT;
    typename T::GradientT;
    typename T::AccumulatorT;
    typename T::OptimizerStateT;
    typename T::LossT;
} && std::is_arithmetic_v<typename T::ParameterT> &&
    std::is_arithmetic_v<typename T::ActivationT> &&
    std::is_arithmetic_v<typename T::GradientT> &&
    std::is_arithmetic_v<typename T::AccumulatorT> &&
    std::is_arithmetic_v<typename T::OptimizerStateT> &&
    std::is_arithmetic_v<typename T::LossT>;

} // namespace edge
