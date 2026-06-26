#pragma once

#include <cstddef>
#include <cstdint>

#include <edge/activations.hpp>
#include <edge/tensor.hpp>

#if defined(__has_include)
#if __has_include(<arm_mve.h>) && defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE > 0) && \
    (defined(__ARM_FEATURE_MVE_FLOAT) || defined(ARM_MATH_MVEF) || defined(ARM_MATH_HELIUM))
#include <arm_mve.h>
#define EDGE_DETAIL_HAS_M55_MVE 1
#endif
#endif

#ifndef EDGE_DETAIL_HAS_M55_MVE
#define EDGE_DETAIL_HAS_M55_MVE 0
#endif

namespace edge::detail {

inline constexpr bool m55_mve_available = EDGE_DETAIL_HAS_M55_MVE != 0;

#if EDGE_DETAIL_HAS_M55_MVE
inline float m55_reduce_add_f32(float32x4_t value) noexcept {
    return vgetq_lane_f32(value, 0) + vgetq_lane_f32(value, 1) +
           vgetq_lane_f32(value, 2) + vgetq_lane_f32(value, 3);
}

inline float m55_dot_f32(const float* lhs, const float* rhs, std::size_t count) noexcept {
    float32x4_t acc = vdupq_n_f32(0.0F);
    std::size_t i = 0;
    for (; i + 4U <= count; i += 4U) {
        const float32x4_t x = vldrwq_f32(lhs + i);
        const float32x4_t w = vldrwq_f32(rhs + i);
        acc = vfmaq_f32(acc, x, w);
    }
    if (i < count) {
        const auto remaining = static_cast<std::uint32_t>(count - i);
        const mve_pred16_t p = vctp32q(remaining);
        const float32x4_t x = vldrwq_z_f32(lhs + i, p);
        const float32x4_t w = vldrwq_z_f32(rhs + i, p);
        acc = vfmaq_m_f32(acc, x, w, p);
    }
    return m55_reduce_add_f32(acc);
}

inline void m55_fill_zero_f32(float* dst, std::size_t count) noexcept {
    std::size_t i = 0;
    const float32x4_t zero = vdupq_n_f32(0.0F);
    for (; i + 4U <= count; i += 4U) {
        vstrwq_f32(dst + i, zero);
    }
    if (i < count) {
        const auto remaining = static_cast<std::uint32_t>(count - i);
        const mve_pred16_t p = vctp32q(remaining);
        vstrwq_p_f32(dst + i, zero, p);
    }
}

inline void m55_axpy_f32(float* dst, const float* src, float alpha, std::size_t count) noexcept {
    std::size_t i = 0;
    const float32x4_t alpha_v = vdupq_n_f32(alpha);
    for (; i + 4U <= count; i += 4U) {
        float32x4_t y = vldrwq_f32(dst + i);
        const float32x4_t x = vldrwq_f32(src + i);
        y = vfmaq_f32(y, x, alpha_v);
        vstrwq_f32(dst + i, y);
    }
    if (i < count) {
        const auto remaining = static_cast<std::uint32_t>(count - i);
        const mve_pred16_t p = vctp32q(remaining);
        float32x4_t y = vldrwq_z_f32(dst + i, p);
        const float32x4_t x = vldrwq_z_f32(src + i, p);
        y = vfmaq_m_f32(y, x, alpha_v, p);
        vstrwq_p_f32(dst + i, y, p);
    }
}
#endif

template<std::size_t InFeatures,
         std::size_t OutFeatures,
         std::size_t CacheCount,
         typename Activation>
bool m55_dense_forward(TensorView<const float, InFeatures> input,
                       TensorView<float, OutFeatures> output,
                       TensorView<const float, InFeatures * OutFeatures + OutFeatures> params,
                       TensorView<float, CacheCount> cache) noexcept {
#if EDGE_DETAIL_HAS_M55_MVE
    constexpr std::size_t weight_count = InFeatures * OutFeatures;
    const float* weights = params.data();
    const float* bias = params.data() + weight_count;

    constexpr std::size_t vector_width = 4U;
    constexpr std::size_t out_blocks = OutFeatures / vector_width;
    std::size_t out = 0;
    for (std::size_t out_block = 0; out_block < out_blocks; ++out_block) {
        out = out_block * vector_width;
        float32x4_t acc0 = vdupq_n_f32(0.0F);
        float32x4_t acc1 = vdupq_n_f32(0.0F);
        float32x4_t acc2 = vdupq_n_f32(0.0F);
        float32x4_t acc3 = vdupq_n_f32(0.0F);

        const float* w0 = weights + (out * InFeatures);
        const float* w1 = w0 + InFeatures;
        const float* w2 = w1 + InFeatures;
        const float* w3 = w2 + InFeatures;

        constexpr std::size_t in_blocks = InFeatures / vector_width;
        std::size_t in = 0;
        for (std::size_t in_block = 0; in_block < in_blocks; ++in_block) {
            in = in_block * vector_width;
            const float32x4_t x = vldrwq_f32(input.data() + in);
            acc0 = vfmaq_f32(acc0, x, vldrwq_f32(w0 + in));
            acc1 = vfmaq_f32(acc1, x, vldrwq_f32(w1 + in));
            acc2 = vfmaq_f32(acc2, x, vldrwq_f32(w2 + in));
            acc3 = vfmaq_f32(acc3, x, vldrwq_f32(w3 + in));
        }
        in = in_blocks * vector_width;

        if (in < InFeatures) {
            const auto remaining = static_cast<std::uint32_t>(InFeatures - in);
            const mve_pred16_t p = vctp32q(remaining);
            const float32x4_t x = vldrwq_z_f32(input.data() + in, p);
            acc0 = vfmaq_m_f32(acc0, x, vldrwq_z_f32(w0 + in, p), p);
            acc1 = vfmaq_m_f32(acc1, x, vldrwq_z_f32(w1 + in, p), p);
            acc2 = vfmaq_m_f32(acc2, x, vldrwq_z_f32(w2 + in, p), p);
            acc3 = vfmaq_m_f32(acc3, x, vldrwq_z_f32(w3 + in, p), p);
        }

        const float z0 = m55_reduce_add_f32(acc0) + bias[out];
        const float z1 = m55_reduce_add_f32(acc1) + bias[out + 1U];
        const float z2 = m55_reduce_add_f32(acc2) + bias[out + 2U];
        const float z3 = m55_reduce_add_f32(acc3) + bias[out + 3U];
        if constexpr (activation_stores_preactivation<Activation, float>()) {
            cache[out] = z0;
            cache[out + 1U] = z1;
            cache[out + 2U] = z2;
            cache[out + 3U] = z3;
        }
        output[out] = Activation::template forward<float>(z0);
        output[out + 1U] = Activation::template forward<float>(z1);
        output[out + 2U] = Activation::template forward<float>(z2);
        output[out + 3U] = Activation::template forward<float>(z3);
    }
    out = out_blocks * vector_width;

    for (; out < OutFeatures; ++out) {
        const float z = m55_dot_f32(input.data(), weights + out * InFeatures, InFeatures) +
                        bias[out];
        if constexpr (activation_stores_preactivation<Activation, float>()) {
            cache[out] = z;
        }
        output[out] = Activation::template forward<float>(z);
    }
    return true;
#else
    (void)input;
    (void)output;
    (void)params;
    (void)cache;
    return false;
#endif
}

template<std::size_t InFeatures,
         std::size_t OutFeatures,
         std::size_t CacheCount,
         typename Activation>
bool m55_dense_backward(
    TensorView<const float, InFeatures> input,
    TensorView<const float, OutFeatures> output,
    TensorView<const float, OutFeatures> upstream,
    TensorView<float, InFeatures> downstream,
    TensorView<const float, InFeatures * OutFeatures + OutFeatures> params,
    TensorView<float, InFeatures * OutFeatures + OutFeatures> gradients,
    TensorView<const float, CacheCount> cache) noexcept {
#if EDGE_DETAIL_HAS_M55_MVE
    constexpr std::size_t weight_count = InFeatures * OutFeatures;
    const float* weights = params.data();
    float* grad_weights = gradients.data();
    float* grad_bias = gradients.data() + weight_count;

    m55_fill_zero_f32(downstream.data(), InFeatures);

    for (std::size_t out = 0; out < OutFeatures; ++out) {
        const float z =
            activation_stores_preactivation<Activation, float>() ? cache[out] : 0.0F;
        const float deriv = activation_derivative<Activation, float>(z, output[out]);
        const float delta = upstream[out] * deriv;
        if (delta == 0.0F) {
            continue;
        }
        const std::size_t row = out * InFeatures;
        grad_bias[out] += delta;
        m55_axpy_f32(grad_weights + row, input.data(), delta, InFeatures);
        m55_axpy_f32(downstream.data(), weights + row, delta, InFeatures);
    }
    return true;
#else
    (void)input;
    (void)output;
    (void)upstream;
    (void)downstream;
    (void)params;
    (void)gradients;
    (void)cache;
    return false;
#endif
}

template<std::size_t InFeatures,
         std::size_t OutFeatures,
         std::size_t CacheCount,
         typename Activation>
bool m55_dense_backward_inputless(
    TensorView<const float, InFeatures> input,
    TensorView<const float, OutFeatures> output,
    TensorView<const float, OutFeatures> upstream,
    TensorView<float, InFeatures * OutFeatures + OutFeatures> gradients,
    TensorView<const float, CacheCount> cache) noexcept {
#if EDGE_DETAIL_HAS_M55_MVE
    constexpr std::size_t weight_count = InFeatures * OutFeatures;
    float* grad_weights = gradients.data();
    float* grad_bias = gradients.data() + weight_count;

    for (std::size_t out = 0; out < OutFeatures; ++out) {
        const float z =
            activation_stores_preactivation<Activation, float>() ? cache[out] : 0.0F;
        const float deriv = activation_derivative<Activation, float>(z, output[out]);
        const float delta = upstream[out] * deriv;
        if (delta == 0.0F) {
            continue;
        }
        const std::size_t row = out * InFeatures;
        grad_bias[out] += delta;
        m55_axpy_f32(grad_weights + row, input.data(), delta, InFeatures);
    }
    return true;
#else
    (void)input;
    (void)output;
    (void)upstream;
    (void)gradients;
    (void)cache;
    return false;
#endif
}

} // namespace edge::detail
