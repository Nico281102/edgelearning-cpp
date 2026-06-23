#pragma once

#include <cmath>

namespace edge {

enum class ActivationStorage {
    None,
    OutputOnly,
    PreActivationOnly,
    OutputAndPreActivation
};

struct Linear {
    static constexpr ActivationStorage storage = ActivationStorage::OutputOnly;

    template<typename T>
    static constexpr T forward(T z) noexcept {
        return z;
    }

    template<typename T>
    static constexpr T derivative_from_output(T) noexcept {
        return T{1};
    }
};

struct ReLU {
    static constexpr ActivationStorage storage = ActivationStorage::OutputOnly;

    template<typename T>
    static constexpr T forward(T z) noexcept {
        return z > T{0} ? z : T{0};
    }

    template<typename T>
    static constexpr T derivative_from_output(T a) noexcept {
        return a > T{0} ? T{1} : T{0};
    }
};

struct Tanh {
    static constexpr ActivationStorage storage = ActivationStorage::OutputOnly;

    template<typename T>
    static T forward(T z) noexcept {
        using std::tanh;
        return static_cast<T>(tanh(z));
    }

    template<typename T>
    static constexpr T derivative_from_output(T a) noexcept {
        return T{1} - a * a;
    }
};

struct Sigmoid {
    static constexpr ActivationStorage storage = ActivationStorage::OutputOnly;

    template<typename T>
    static T forward(T z) noexcept {
        using std::exp;
        return T{1} / (T{1} + static_cast<T>(exp(-z)));
    }

    template<typename T>
    static constexpr T derivative_from_output(T a) noexcept {
        return a * (T{1} - a);
    }
};

template<typename Activation, typename T>
constexpr bool activation_stores_preactivation() noexcept {
    return Activation::storage == ActivationStorage::PreActivationOnly ||
           Activation::storage == ActivationStorage::OutputAndPreActivation;
}

template<typename Activation, typename T>
T activation_derivative(T z, T a) noexcept {
    if constexpr (requires { Activation::template derivative_from_output<T>(a); }) {
        return Activation::template derivative_from_output<T>(a);
    } else if constexpr (requires { Activation::template derivative<T>(z, a); }) {
        return Activation::template derivative<T>(z, a);
    } else if constexpr (requires { Activation::template derivative_from_preactivation<T>(z); }) {
        return Activation::template derivative_from_preactivation<T>(z);
    } else {
        static_assert(sizeof(Activation) == 0,
                      "Activation must provide derivative_from_output(a), derivative(z, a), or derivative_from_preactivation(z)");
    }
}

} // namespace edge

