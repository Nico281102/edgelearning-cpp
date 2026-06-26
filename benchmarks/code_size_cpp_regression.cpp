#include <array>

#include <edge/edge.hpp>

namespace {

using Model = edge::Model<
    edge::InputVector<1>,
    edge::Dense<4, edge::Tanh>,
    edge::Dense<1, edge::Linear>>;

volatile float sink = 0.0F;

} // namespace

int main() {
    edge::Trainer<Model, edge::MSE, edge::SGD> trainer(
        edge::TrainerConfig{.batch_size = 4, .reduction = edge::GradientReduction::Mean},
        edge::SGDConfig{.learning_rate = 0.05F});
    trainer.model().initialize(edge::InitConfig{.seed = 1U});

    const std::array<float, 1> input{1.0F};
    const std::array<float, 1> target{3.0F};
    const edge::Status status = trainer.train_step(input, target);
    trainer.flush();
    sink = trainer.last_loss() + trainer.model().output()[0];

    return edge::is_ok(status) ? 0 : 1;
}
