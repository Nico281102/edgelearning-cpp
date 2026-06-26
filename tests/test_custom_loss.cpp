#include <array>
#include <cmath>

#include <edge/edge.hpp>

#include "test_harness.hpp"

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
    using Model = edge::Model<edge::InputVector<1>, edge::Dense<1>>;
    edge::Trainer<Model, L1Loss, edge::SGD> trainer(edge::SGDConfig{.learning_rate = 0.1F});
    trainer.model().parameter_data()[0] = 0.0F;
    trainer.model().parameter_data()[1] = 0.0F;

    const std::array<float, 1> input{2.0F};
    const std::array<float, 1> target{3.0F};
    EDGE_EXPECT_EQ(trainer.train_step(input, target), edge::Status::Ok);
    EDGE_EXPECT_NEAR(trainer.last_loss(), 3.0F, 1.0e-6F);
    EDGE_EXPECT_NEAR(trainer.model().parameter_data()[0], 0.2F, 1.0e-6F);
}

