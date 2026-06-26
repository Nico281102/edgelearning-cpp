#include <array>

#include <edge/edge.hpp>

#include "test_harness.hpp"

template<typename Model>
void expect_same_parameters(const Model& a, const Model& b) {
    for (std::size_t i = 0; i < Model::parameter_count; ++i) {
        EDGE_EXPECT_NEAR(a.parameter_data()[i], b.parameter_data()[i], 1.0e-7F);
    }
}

int main() {
    {
        using Model = edge::Model<edge::InputVector<2>, edge::Dense<3, edge::ReLU, edge::XavierUniform>>;
        Model a;
        Model b;
        EDGE_EXPECT_EQ(a.initialize(edge::InitConfig{.seed = 42U}), edge::Status::Ok);
        EDGE_EXPECT_EQ(b.initialize(edge::InitConfig{.seed = 42U}), edge::Status::Ok);
        expect_same_parameters(a, b);
        const float* p = a.parameter_data();
        EDGE_EXPECT_NEAR(p[6], 0.0F, 1.0e-7F);
        EDGE_EXPECT_NEAR(p[7], 0.0F, 1.0e-7F);
        EDGE_EXPECT_NEAR(p[8], 0.0F, 1.0e-7F);
    }

    {
        using Model = edge::Model<edge::InputVector<2>, edge::Dense<2, edge::Linear, edge::KaimingUniform>>;
        Model model;
        EDGE_EXPECT_EQ(model.initialize(edge::InitConfig{.seed = 7U, .bias = 0.25F}),
                       edge::Status::Ok);
        EDGE_EXPECT_NEAR(model.parameter_data()[4], 0.25F, 1.0e-7F);
        EDGE_EXPECT_NEAR(model.parameter_data()[5], 0.25F, 1.0e-7F);
    }

    {
        using Model = edge::Model<edge::InputVector<3>, edge::Dense<2, edge::Linear, edge::Orthogonal>>;
        Model model;
        EDGE_EXPECT_EQ(model.initialize(edge::InitConfig{.gain = 2.0F}), edge::Status::Ok);
        EDGE_EXPECT_NEAR(model.parameter_data()[0], 2.0F, 1.0e-7F);
        EDGE_EXPECT_NEAR(model.parameter_data()[4], 2.0F, 1.0e-7F);
    }

    {
        using Model = edge::Model<edge::InputVector<2>, edge::Dense<1, edge::Linear, edge::Constant>>;
        Model model;
        EDGE_EXPECT_EQ(model.initialize(edge::InitConfig{.constant = -0.5F}), edge::Status::Ok);
        EDGE_EXPECT_NEAR(model.parameter_data()[0], -0.5F, 1.0e-7F);
        EDGE_EXPECT_NEAR(model.parameter_data()[1], -0.5F, 1.0e-7F);
    }
}

