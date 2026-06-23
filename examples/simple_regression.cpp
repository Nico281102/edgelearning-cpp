#include <array>

#include <edge/edge.hpp>

int main() {
    using Model = edge::Model<
        edge::Input<8>,
        edge::Dense<32, edge::ReLU>,
        edge::Dense<1>>;

    edge::Trainer<Model, edge::MSE, edge::Adam> trainer(edge::AdamConfig{.learning_rate = 1.0e-3F});
    trainer.model().initialize(edge::InitConfig{.seed = 42U});

    std::array<float, 8> input{0.0F, 1.0F, 0.5F, -0.5F, 0.25F, 0.75F, -1.0F, 0.1F};
    std::array<float, 1> target{0.25F};
    return edge::is_ok(trainer.train_step(input, target)) ? 0 : 1;
}

