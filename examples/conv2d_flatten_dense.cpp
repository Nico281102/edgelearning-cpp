#include <array>
#include <cstddef>

#include <edge/edge.hpp>

int main() {
    using Model = edge::Model<
        edge::Input<edge::CHW<1, 28, 28>>,
        edge::Conv2D<4,
                     edge::Kernel<3, 3>,
                     edge::ReLU,
                     edge::DefaultInitializer,
                     edge::Stride<1, 1>,
                     edge::Padding<1, 1>>,
        edge::Flatten,
        edge::Dense<10>>;

    edge::Trainer<Model, edge::MSE, edge::SGD> trainer(
        edge::SGDConfig{.learning_rate = 1.0e-3F});
    trainer.model().initialize(edge::InitConfig{.seed = 28U});

    std::array<float, Model::input_size> image{};
    for (std::size_t i = 0; i < image.size(); ++i) {
        image[i] = static_cast<float>((i * 13U) % 256U) / 255.0F;
    }

    std::array<float, Model::output_size> target{};
    target[3] = 1.0F;

    return edge::is_ok(trainer.train_step(image, target)) ? 0 : 1;
}
