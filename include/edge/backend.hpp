#pragma once

#include <type_traits>

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
struct is_backend_policy : std::false_type {};

template<>
struct is_backend_policy<Backend::Generic> : std::true_type {};

template<>
struct is_backend_policy<Backend::M55> : std::true_type {};

} // namespace detail

template<typename T>
concept BackendPolicy = detail::is_backend_policy<T>::value;

} // namespace edge

