#pragma once

#include <cstddef>

namespace edge {

struct MSE {
    template<typename Prediction, typename Target>
    static float value(const Prediction& prediction, const Target& target) noexcept {
        static_assert(Prediction::extent == Target::extent,
                      "MSE prediction and target extents must match");
        float sum = 0.0F;
        for (std::size_t i = 0; i < Prediction::extent; ++i) {
            const float diff = prediction[i] - target[i];
            sum += diff * diff;
        }
        return sum / static_cast<float>(Prediction::extent);
    }

    template<typename Prediction, typename Target, typename Gradient>
    static float evaluate(const Prediction& prediction,
                          const Target& target,
                          Gradient& gradient) noexcept {
        static_assert(Prediction::extent == Target::extent,
                      "MSE prediction and target extents must match");
        static_assert(Prediction::extent == Gradient::extent,
                      "MSE gradient extent must match prediction extent");
        float sum = 0.0F;
        const float scale = 2.0F / static_cast<float>(Prediction::extent);
        for (std::size_t i = 0; i < Prediction::extent; ++i) {
            const float diff = prediction[i] - target[i];
            sum += diff * diff;
            gradient[i] = scale * diff;
        }
        return sum / static_cast<float>(Prediction::extent);
    }
};

} // namespace edge

