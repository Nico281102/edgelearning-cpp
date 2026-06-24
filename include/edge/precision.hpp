#pragma once

#include <concepts>
#include <type_traits>

namespace edge {

struct UseModelPrecision {};

namespace precision {

#if defined(__FLT16_MANT_DIG__)
using Float16Storage = _Float16;
inline constexpr bool has_native_float16_storage = true;
#else
using Float16Storage = float;
inline constexpr bool has_native_float16_storage = false;
#endif

struct FP32 {
    using ParameterT = float;
    using ActivationT = float;
    using GradientT = float;
    using AccumulatorT = float;
    using OptimizerStateT = float;
    using LossT = float;
};

struct MixedFP16 {
    // Mixed precision policy inspired by FP16 activation/storage paths with FP32
    // master parameters and FP32 accumulation. On compilers without _Float16,
    // ActivationT falls back to float and has_native_float16_storage is false.
    using ParameterT = float;
    using ActivationT = Float16Storage;
    using GradientT = float;
    using AccumulatorT = float;
    using OptimizerStateT = float;
    using LossT = float;
    static constexpr bool uses_native_float16_storage = has_native_float16_storage;
};

} // namespace precision

namespace detail {

template<typename T>
concept PrecisionScalar = std::is_arithmetic_v<T> || requires(T value) {
    { static_cast<T>(0.0F) };
    { static_cast<float>(value) } -> std::convertible_to<float>;
};

} // namespace detail

template<typename T>
concept PrecisionPolicy = requires {
    typename T::ParameterT;
    typename T::ActivationT;
    typename T::GradientT;
    typename T::AccumulatorT;
    typename T::OptimizerStateT;
    typename T::LossT;
} && detail::PrecisionScalar<typename T::ParameterT> &&
    detail::PrecisionScalar<typename T::ActivationT> &&
    detail::PrecisionScalar<typename T::GradientT> &&
    detail::PrecisionScalar<typename T::AccumulatorT> &&
    detail::PrecisionScalar<typename T::OptimizerStateT> &&
    detail::PrecisionScalar<typename T::LossT>;

} // namespace edge
