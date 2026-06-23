#include <array>
#include <cstddef>

#include <edge/edge.hpp>

#include "test_harness.hpp"

int main() {
    using Model = edge::Model<
        edge::Input<28 * 28>,
        edge::Conv2D<1, 28, 28, 4, 3, 3, edge::ReLU, edge::DefaultInitializer, 1, 1, 1, 1>,
        edge::Dense<10, edge::Linear>>;

    static_assert(Model::input_size == 28 * 28);
    static_assert(Model::output_size == 10);
    static_assert(Model::parameter_count == (4 * 1 * 3 * 3 + 4 + 10 * 4 * 28 * 28 + 10));

    edge::Trainer<Model, edge::MSE, edge::SGD> trainer(
        edge::SGDConfig{.learning_rate = 1.0e-3F});
    EDGE_EXPECT_EQ(trainer.model().initialize(edge::InitConfig{.seed = 28U}), edge::Status::Ok);

    std::array<float, Model::input_size> image{};
    for (std::size_t i = 0; i < image.size(); ++i) {
        const auto pixel = static_cast<float>((i * 17U + 11U) % 256U);
        image[i] = pixel / 255.0F;
    }

    std::array<float, Model::output_size> target{};
    target[7] = 1.0F;

    EDGE_EXPECT_EQ(trainer.train_step(image, target), edge::Status::Ok);
    EDGE_EXPECT_TRUE(trainer.last_loss() >= 0.0F);
}
