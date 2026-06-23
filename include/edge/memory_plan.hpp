#pragma once

#include <cstddef>

namespace edge {

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
    return ((value + alignment - 1U) / alignment) * alignment;
}

struct MemoryBreakdown {
    std::size_t parameter_bytes = 0;
    std::size_t gradient_bytes = 0;
    std::size_t optimizer_bytes = 0;
    std::size_t activation_bytes = 0;
    std::size_t workspace_bytes = 0;
    std::size_t total_bytes = 0;
};

} // namespace edge

