#include <array>

#include <edge/edge.hpp>

#include "test_harness.hpp"

int main() {
    {
        using Model = edge::Model<
            edge::Input<4>,
            edge::Conv2D<1, 2, 2, 1, 2, 2, edge::Linear>>;

        static_assert(Model::input_size == 4);
        static_assert(Model::output_size == 1);
        static_assert(Model::parameter_count == 5);

        Model model;
        float* p = model.parameter_data();
        p[0] = 0.5F;
        p[1] = -1.0F;
        p[2] = 0.25F;
        p[3] = 2.0F;
        p[4] = 0.1F;

        const std::array<float, 4> input{1.0F, 2.0F, 3.0F, 4.0F};
        const std::array<float, 1> upstream{2.0F};

        EDGE_EXPECT_EQ(model.forward(input), edge::Status::Ok);
        EDGE_EXPECT_NEAR(model.output()[0], 7.35F, 1.0e-6F);

        EDGE_EXPECT_EQ(model.zero_grad(), edge::Status::Ok);
        EDGE_EXPECT_EQ(model.backward(upstream), edge::Status::Ok);

        const float* g = model.gradient_data();
        EDGE_EXPECT_NEAR(g[0], 2.0F, 1.0e-6F);
        EDGE_EXPECT_NEAR(g[1], 4.0F, 1.0e-6F);
        EDGE_EXPECT_NEAR(g[2], 6.0F, 1.0e-6F);
        EDGE_EXPECT_NEAR(g[3], 8.0F, 1.0e-6F);
        EDGE_EXPECT_NEAR(g[4], 2.0F, 1.0e-6F);
    }

    {
        using Model = edge::Model<
            edge::Input<9>,
            edge::Conv2D<1, 3, 3, 1, 2, 2, edge::Linear, edge::Constant, 2, 2, 1, 1>>;

        static_assert(Model::output_size == 4);

        Model model;
        float* p = model.parameter_data();
        for (std::size_t i = 0; i < 4; ++i) {
            p[i] = 1.0F;
        }
        p[4] = 0.0F;

        const std::array<float, 9> input{
            1.0F, 2.0F, 3.0F,
            4.0F, 5.0F, 6.0F,
            7.0F, 8.0F, 9.0F};

        EDGE_EXPECT_EQ(model.forward(input), edge::Status::Ok);
        auto out = model.output();
        EDGE_EXPECT_NEAR(out[0], 1.0F, 1.0e-6F);
        EDGE_EXPECT_NEAR(out[1], 5.0F, 1.0e-6F);
        EDGE_EXPECT_NEAR(out[2], 11.0F, 1.0e-6F);
        EDGE_EXPECT_NEAR(out[3], 28.0F, 1.0e-6F);
    }
}
