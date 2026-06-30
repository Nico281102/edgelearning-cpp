#include <array>
#include <cstddef>

#include <edge/edge.hpp>

namespace {

constexpr std::size_t kInputSize = 3;
constexpr std::size_t kBatchSize = 256;
constexpr std::size_t kSamples = 2 * kBatchSize + 17;

std::array<float, kInputSize> make_input(std::size_t sample) {
    const float t = static_cast<float>(sample % 31U) / 30.0F;
    return {t, 1.0F - t, 0.5F * t};
}

std::array<float, 1> make_target(const std::array<float, kInputSize>& input) {
    return {0.2F * input[0] - 0.4F * input[1] + 0.1F * input[2]};
}

} // namespace

int main() {
    using Model = edge::Model<
        edge::InputVector<kInputSize>,
        edge::Dense<8, edge::ReLU>,
        edge::Dense<1>>;

    edge::Trainer<Model, edge::MSE, edge::Adam> trainer(
        edge::TrainerConfig{
            .batch_size = kBatchSize,
            .reduction = edge::GradientReduction::Mean,
        },
        edge::AdamConfig{.learning_rate = 1.0e-3F});
    trainer.model().initialize(edge::InitConfig{.seed = 42U});

    for (std::size_t sample = 0; sample < kSamples; ++sample) {
        const auto input = make_input(sample);
        const auto target = make_target(input);
        if (!edge::is_ok(trainer.train_step(input, target))) {
            return 1;
        }
    }

    if (!edge::is_ok(trainer.flush())) {
        return 1;
    }
    return trainer.accumulated_samples() == 0U ? 0 : 1;
}
