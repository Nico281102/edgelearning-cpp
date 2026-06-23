#include <array>
#include <cmath>

#include <edge/edge.hpp>

#include "test_harness.hpp"

int main() {
    {
        using Model = edge::Model<edge::Input<2>, edge::Dense<2, edge::Linear>>;
        Model model;
        float* p = model.parameter_data();
        p[0] = 1.0F;
        p[1] = 2.0F;
        p[2] = 3.0F;
        p[3] = 4.0F;
        p[4] = 0.5F;
        p[5] = -0.5F;
        const std::array<float, 2> input{2.0F, 1.0F};
        EDGE_EXPECT_EQ(model.forward(input), edge::Status::Ok);
        auto out = model.output();
        EDGE_EXPECT_NEAR(out[0], 4.5F, 1.0e-6F);
        EDGE_EXPECT_NEAR(out[1], 9.5F, 1.0e-6F);
    }

    {
        using Model = edge::Model<edge::Input<2>, edge::Dense<2, edge::ReLU>>;
        Model model;
        float* p = model.parameter_data();
        p[0] = 1.0F;
        p[1] = -3.0F;
        p[2] = -2.0F;
        p[3] = 1.0F;
        p[4] = 0.0F;
        p[5] = 0.0F;
        const std::array<float, 2> input{1.0F, 1.0F};
        EDGE_EXPECT_EQ(model.forward(input), edge::Status::Ok);
        auto out = model.output();
        EDGE_EXPECT_NEAR(out[0], 0.0F, 1.0e-6F);
        EDGE_EXPECT_NEAR(out[1], 0.0F, 1.0e-6F);
    }

    {
        using Model = edge::Model<edge::Input<1>, edge::Dense<1, edge::Tanh>>;
        Model model;
        float* p = model.parameter_data();
        p[0] = 2.0F;
        p[1] = 0.0F;
        const std::array<float, 1> input{0.25F};
        EDGE_EXPECT_EQ(model.forward(input), edge::Status::Ok);
        EDGE_EXPECT_NEAR(model.output()[0], std::tanh(0.5F), 1.0e-6F);
    }

    {
        using Model = edge::Model<edge::Input<1>, edge::Dense<1, edge::Sigmoid>>;
        Model model;
        float* p = model.parameter_data();
        p[0] = 4.0F;
        p[1] = 0.0F;
        const std::array<float, 1> input{0.0F};
        EDGE_EXPECT_EQ(model.forward(input), edge::Status::Ok);
        EDGE_EXPECT_NEAR(model.output()[0], 0.5F, 1.0e-6F);
    }
}

