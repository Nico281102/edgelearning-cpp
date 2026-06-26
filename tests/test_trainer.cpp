#include <array>

#include <edge/edge.hpp>

#include "test_harness.hpp"

namespace {

using Model = edge::Model<edge::InputVector<1>, edge::Dense<1, edge::Linear>>;

float train_two_samples(edge::GradientReduction reduction, std::size_t batch_size) {
    edge::Trainer<Model, edge::MSE, edge::SGD> trainer(
        edge::TrainerConfig{.batch_size = batch_size, .reduction = reduction},
        edge::SGDConfig{.learning_rate = 0.1F});
    trainer.model().parameter_data()[0] = 0.0F;
    trainer.model().parameter_data()[1] = 0.0F;

    const std::array<float, 1> x1{1.0F};
    const std::array<float, 1> y1{1.0F};
    const std::array<float, 1> x2{2.0F};
    const std::array<float, 1> y2{2.0F};

    EDGE_EXPECT_EQ(trainer.train_step(x1, y1), edge::Status::Ok);
    EDGE_EXPECT_EQ(trainer.train_step(x2, y2), edge::Status::Ok);
    EDGE_EXPECT_EQ(trainer.flush(), edge::Status::Ok);
    return trainer.model().parameter_data()[0];
}

} // namespace

int main() {
    EDGE_EXPECT_NEAR(train_two_samples(edge::GradientReduction::Mean, 2), 0.5F, 1.0e-6F);
    EDGE_EXPECT_NEAR(train_two_samples(edge::GradientReduction::Sum, 2), 1.0F, 1.0e-6F);
    EDGE_EXPECT_NEAR(train_two_samples(edge::GradientReduction::Mean, 3), 0.5F, 1.0e-6F);
}

