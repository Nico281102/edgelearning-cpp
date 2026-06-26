#pragma once

#include <concepts>

namespace edge {

struct Backend {
    struct Generic {
        static constexpr bool is_backend_policy = true;
        static constexpr const char* name = "Generic";
    };

    struct M55 {
        static constexpr bool is_backend_policy = true;
        static constexpr const char* name = "M55";
        static constexpr bool falls_back_to_generic = true;
    };

    using Default = Generic;
};

namespace detail {

template<typename T>
consteval bool backend_falls_back_to_generic() {
    if constexpr (requires { T::falls_back_to_generic; }) {
        return T::falls_back_to_generic;
    } else {
        return true;
    }
}

template<typename>
inline constexpr bool always_false_v = false;

} // namespace detail

template<typename T>
concept BackendPolicy = requires {
    { T::is_backend_policy } -> std::convertible_to<bool>;
} && T::is_backend_policy;

template<typename T>
inline constexpr bool backend_falls_back_to_generic_v =
    detail::backend_falls_back_to_generic<T>();

} // namespace edge
