#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace edge {

struct InitConfig {
    std::uint32_t seed = 1U;
    float bias = 0.0F;
    float gain = 1.0F;
    float constant = 0.0F;
};

class DeterministicRng {
public:
    explicit constexpr DeterministicRng(std::uint32_t seed) noexcept
        : state_(seed == 0U ? 1U : seed) {}

    constexpr std::uint32_t next_u32() noexcept {
        state_ = state_ * 1664525U + 1013904223U;
        return state_;
    }

    float uniform(float low, float high) noexcept {
        const float unit = static_cast<float>(next_u32() >> 8U) *
                           (1.0F / static_cast<float>(1U << 24U));
        return low + (high - low) * unit;
    }

private:
    std::uint32_t state_;
};

struct XavierUniform {
    template<typename T, std::size_t In, std::size_t Out>
    static void fill(T* weights, DeterministicRng& rng, const InitConfig& config) noexcept {
        const float limit = config.gain * std::sqrt(6.0F / static_cast<float>(In + Out));
        for (std::size_t i = 0; i < In * Out; ++i) {
            weights[i] = static_cast<T>(rng.uniform(-limit, limit));
        }
    }
};

struct KaimingUniform {
    template<typename T, std::size_t In, std::size_t Out>
    static void fill(T* weights, DeterministicRng& rng, const InitConfig& config) noexcept {
        const float limit = config.gain * std::sqrt(6.0F / static_cast<float>(In));
        for (std::size_t i = 0; i < In * Out; ++i) {
            weights[i] = static_cast<T>(rng.uniform(-limit, limit));
        }
    }
};

struct Orthogonal {
    template<typename T, std::size_t In, std::size_t Out>
    static void fill(T* weights, DeterministicRng&, const InitConfig& config) noexcept {
        for (std::size_t i = 0; i < In * Out; ++i) {
            weights[i] = T{0};
        }
        const std::size_t diagonal = In < Out ? In : Out;
        for (std::size_t i = 0; i < diagonal; ++i) {
            weights[i * In + i] = static_cast<T>(config.gain);
        }
    }
};

struct Constant {
    template<typename T, std::size_t In, std::size_t Out>
    static void fill(T* weights, DeterministicRng&, const InitConfig& config) noexcept {
        for (std::size_t i = 0; i < In * Out; ++i) {
            weights[i] = static_cast<T>(config.constant);
        }
    }
};

struct DefaultInitializer : XavierUniform {};

} // namespace edge
