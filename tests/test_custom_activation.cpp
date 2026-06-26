#include <array>

#include <edge/edge.hpp>

#include "test_harness.hpp"

struct SquareActivation {
    static constexpr edge::ActivationStorage storage =
        edge::ActivationStorage::OutputAndPreActivation;

    template<typename T>
    static constexpr T forward(T z) noexcept {
        return z * z;
    }

    template<typename T>
    static constexpr T derivative(T z, T) noexcept {
        return T{2} * z;
    }
};

int main() {
    using Model = edge::Model<edge::InputVector<1>, edge::Dense<1, SquareActivation>>;
    static_assert(Model::preactivation_count == 1);

    Model model;
    model.parameter_data()[0] = 3.0F;
    model.parameter_data()[1] = 1.0F;
    const std::array<float, 1> input{2.0F};
    const std::array<float, 1> upstream{1.0F};

    EDGE_EXPECT_EQ(model.zero_grad(), edge::Status::Ok);
    EDGE_EXPECT_EQ(model.forward(input), edge::Status::Ok);
    EDGE_EXPECT_NEAR(model.output()[0], 49.0F, 1.0e-6F);
    EDGE_EXPECT_EQ(model.backward(upstream), edge::Status::Ok);
    EDGE_EXPECT_NEAR(model.gradient_data()[0], 28.0F, 1.0e-6F);
    EDGE_EXPECT_NEAR(model.gradient_data()[1], 14.0F, 1.0e-6F);
}

