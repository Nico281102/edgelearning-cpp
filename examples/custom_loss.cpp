#include <array>
#include <cmath>
#include <cstddef>

#include <edge/edge.hpp>

struct L1Loss {
    template<typename Prediction, typename Target>
    static float value(const Prediction& prediction, const Target& target) noexcept {
        float sum = 0.0F;
        for (std::size_t i = 0; i < Prediction::extent; ++i) {
            sum += std::fabs(prediction[i] - target[i]);
        }
        return sum / static_cast<float>(Prediction::extent);
    }

    template<typename Prediction, typename Target, typename Gradient>
    static float evaluate(const Prediction& prediction,
                          const Target& target,
                          Gradient& gradient) noexcept {
        float sum = 0.0F;
        for (std::size_t i = 0; i < Prediction::extent; ++i) {
            const float diff = prediction[i] - target[i];
            sum += std::fabs(diff);
            gradient[i] = diff >= 0.0F ? 1.0F : -1.0F;
        }
        return sum / static_cast<float>(Prediction::extent);
    }
};

int main() {
    using Model = edge::Model<edge::InputVector<2>, edge::Dense<1>>;
    edge::Trainer<Model, L1Loss, edge::SGD> trainer(edge::SGDConfig{.learning_rate = 0.01F});
    trainer.model().initialize(edge::InitConfig{.seed = 1U});

    const std::array<float, 2> input{0.5F, -1.0F};
    const std::array<float, 1> target{0.25F};
    return edge::is_ok(trainer.train_step(input, target)) ? 0 : 1;
}

