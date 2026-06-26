#include <array>

#include <edge/edge.hpp>

#include "test_harness.hpp"

int main() {
    using Model = edge::Model<edge::InputVector<2>, edge::Dense<1, edge::Linear>>;
    Model model;
    float* p = model.parameter_data();
    p[0] = 0.5F;
    p[1] = -1.0F;
    p[2] = 0.0F;

    const std::array<float, 2> input{2.0F, 3.0F};
    const std::array<float, 1> upstream{4.0F};

    EDGE_EXPECT_EQ(model.zero_grad(), edge::Status::Ok);
    EDGE_EXPECT_EQ(model.forward(input), edge::Status::Ok);
    EDGE_EXPECT_EQ(model.backward(upstream), edge::Status::Ok);

    const float* g = model.gradient_data();
    EDGE_EXPECT_NEAR(g[0], 8.0F, 1.0e-6F);
    EDGE_EXPECT_NEAR(g[1], 12.0F, 1.0e-6F);
    EDGE_EXPECT_NEAR(g[2], 4.0F, 1.0e-6F);
}

