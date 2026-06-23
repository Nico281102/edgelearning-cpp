#pragma once

#include <cstddef>
#include <type_traits>

namespace edge {

struct MSE {
    template<typename Prediction, typename Target>
    static auto value(const Prediction& prediction, const Target& target) noexcept {
        static_assert(Prediction::extent == Target::extent,
                      "MSE prediction and target extents must match");
        using LossT = std::common_type_t<
            std::remove_cv_t<typename Prediction::value_type>,
            std::remove_cv_t<typename Target::value_type>,
            float>;
        LossT sum = LossT{0};
        for (std::size_t i = 0; i < Prediction::extent; ++i) {
            const LossT diff =
                static_cast<LossT>(prediction[i]) - static_cast<LossT>(target[i]);
            sum += diff * diff;
        }
        return sum / static_cast<LossT>(Prediction::extent);
    }

    template<typename Prediction, typename Target, typename Gradient>
    static auto evaluate(const Prediction& prediction,
                         const Target& target,
                         Gradient& gradient) noexcept {
        static_assert(Prediction::extent == Target::extent,
                      "MSE prediction and target extents must match");
        static_assert(Prediction::extent == Gradient::extent,
                      "MSE gradient extent must match prediction extent");
        using LossT = std::common_type_t<
            std::remove_cv_t<typename Prediction::value_type>,
            std::remove_cv_t<typename Target::value_type>,
            float>;
        LossT sum = LossT{0};
        const LossT scale = LossT{2} / static_cast<LossT>(Prediction::extent);
        for (std::size_t i = 0; i < Prediction::extent; ++i) {
            const LossT diff =
                static_cast<LossT>(prediction[i]) - static_cast<LossT>(target[i]);
            sum += diff * diff;
            gradient[i] = static_cast<typename Gradient::value_type>(scale * diff);
        }
        return sum / static_cast<LossT>(Prediction::extent);
    }
};

} // namespace edge
