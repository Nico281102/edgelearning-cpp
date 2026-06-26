#include <array>

#include <edge/edge.hpp>

#include "test_harness.hpp"

int main() {
    {
        using Model = edge::Model<edge::InputVector<1>, edge::Dense<1>>;
        Model model;
        model.parameter_data()[0] = 1.0F;
        model.gradient_data()[0] = 2.0F;
        edge::SGD sgd(edge::SGDConfig{.learning_rate = 0.25F});
        EDGE_EXPECT_EQ(sgd.step(model, 1.0F), edge::Status::Ok);
        EDGE_EXPECT_NEAR(model.parameter_data()[0], 0.5F, 1.0e-6F);
        EDGE_EXPECT_EQ(sgd.step_count(), 1U);
    }

    {
        using Model = edge::Model<edge::InputVector<1>, edge::Dense<1>>;
        edge::Trainer<Model, edge::MSE, edge::Adam> a(edge::AdamConfig{.learning_rate = 0.01F});
        edge::Trainer<Model, edge::MSE, edge::Adam> b(edge::AdamConfig{.learning_rate = 0.01F});
        const std::array<float, 1> input{1.0F};
        const std::array<float, 1> target{1.0F};
        EDGE_EXPECT_EQ(a.train_step(input, target), edge::Status::Ok);
        EDGE_EXPECT_EQ(a.optimizer().step_count(), 1U);
        EDGE_EXPECT_EQ(b.optimizer().step_count(), 0U);
    }
}

