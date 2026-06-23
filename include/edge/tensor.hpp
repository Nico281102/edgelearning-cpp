#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

namespace edge {

template<typename T, std::size_t N>
class TensorView {
public:
    using value_type = T;
    static constexpr std::size_t extent = N;

    constexpr TensorView() noexcept = default;

    constexpr explicit TensorView(T* data) noexcept
        : data_(data) {}

    template<typename U>
        requires std::is_convertible_v<U*, T*>
    constexpr TensorView(std::array<U, N>& data) noexcept
        : data_(data.data()) {}

    template<typename U>
        requires std::is_convertible_v<const U*, T*>
    constexpr TensorView(const std::array<U, N>& data) noexcept
        : data_(data.data()) {}

    template<typename U>
        requires std::is_convertible_v<U*, T*>
    constexpr TensorView(U (&data)[N]) noexcept
        : data_(data) {}

    template<typename U>
        requires std::is_convertible_v<const U*, T*>
    constexpr TensorView(const U (&data)[N]) noexcept
        : data_(data) {}

    constexpr T* data() const noexcept {
        return data_;
    }

    constexpr std::size_t size() const noexcept {
        return N;
    }

    constexpr T& operator[](std::size_t index) const noexcept {
        return data_[index];
    }

    constexpr explicit operator bool() const noexcept {
        return data_ != nullptr;
    }

private:
    T* data_ = nullptr;
};

template<typename T, std::size_t N>
struct StaticTensor {
    std::array<T, N> values{};

    constexpr TensorView<T, N> view() noexcept {
        return TensorView<T, N>(values);
    }

    constexpr TensorView<const T, N> view() const noexcept {
        return TensorView<const T, N>(values);
    }

    constexpr T& operator[](std::size_t index) noexcept {
        return values[index];
    }

    constexpr const T& operator[](std::size_t index) const noexcept {
        return values[index];
    }
};

} // namespace edge

