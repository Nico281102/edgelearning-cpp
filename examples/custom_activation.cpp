#include <array>
#include <cmath>

#include <edge/edge.hpp>

struct Swish {
    static constexpr edge::ActivationStorage storage =
        edge::ActivationStorage::OutputAndPreActivation;

    template<typename T>
    static T forward(T z) noexcept {
        const T s = T{1} / (T{1} + static_cast<T>(std::exp(-z)));
        return z * s;
    }

    template<typename T>
    static T derivative(T z, T) noexcept {
        const T s = T{1} / (T{1} + static_cast<T>(std::exp(-z)));
        return s + z * s * (T{1} - s);
    }
};

int main() {
    using Model = edge::Model<edge::InputVector<2>, edge::Dense<4, Swish>, edge::Dense<1>>;
    Model model;
    model.initialize(edge::InitConfig{.seed = 3U});
    const std::array<float, 2> input{0.2F, -0.1F};
    return edge::is_ok(model.forward(input)) ? 0 : 1;
}
