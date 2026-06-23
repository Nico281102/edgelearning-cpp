#pragma once

#include <array>
#include <cstddef>
#include <span>

namespace edge {

template<std::size_t N>
struct ExternalArena {
    std::byte* data;
    static constexpr std::size_t size = N;
};

template<std::size_t N>
constexpr ExternalArena<N> external_arena(std::array<std::byte, N>& arena) noexcept {
    return ExternalArena<N>{arena.data()};
}

template<std::size_t N>
constexpr ExternalArena<N> external_arena(std::byte (&arena)[N]) noexcept {
    return ExternalArena<N>{arena};
}

template<std::size_t N>
constexpr ExternalArena<N> external_arena(std::span<std::byte, N> arena) noexcept {
    return ExternalArena<N>{arena.data()};
}

} // namespace edge

