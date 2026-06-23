#pragma once

#include <cstddef>

namespace edge {

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
    return ((value + alignment - 1U) / alignment) * alignment;
}

template<std::size_t A, std::size_t B, std::size_t... Rest>
struct StaticMax {
    static constexpr std::size_t head = A > B ? A : B;
    static constexpr std::size_t value = StaticMax<head, Rest...>::value;
};

template<std::size_t A, std::size_t B>
struct StaticMax<A, B> {
    static constexpr std::size_t value = A > B ? A : B;
};

template<std::size_t A, std::size_t B, std::size_t... Rest>
inline constexpr std::size_t static_max_v = StaticMax<A, B, Rest...>::value;

struct MemoryBreakdown {
    std::size_t parameter_bytes = 0;
    std::size_t gradient_bytes = 0;
    std::size_t optimizer_bytes = 0;
    std::size_t activation_bytes = 0;
    std::size_t workspace_bytes = 0;
    std::size_t total_bytes = 0;
};

} // namespace edge
