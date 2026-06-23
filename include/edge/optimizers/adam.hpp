#pragma once

#include <cmath>
#include <cstddef>

#include <edge/status.hpp>

namespace edge {

struct AdamConfig {
    float learning_rate = 1.0e-3F;
    float beta1 = 0.9F;
    float beta2 = 0.999F;
    float epsilon = 1.0e-8F;
};

class Adam {
public:
    using Config = AdamConfig;

    explicit constexpr Adam(Config config = {}) noexcept
        : config_(config) {}

    template<typename Model>
    Status step(Model& model, float gradient_scale) noexcept {
        float* params = model.parameter_data();
        const float* grads = model.gradient_data();
        float* state = model.optimizer_state_data();
        if (params == nullptr || grads == nullptr || state == nullptr) {
            return Status::NullPointer;
        }

        ++step_count_;
        float* m = state;
        float* v = state + Model::parameter_count;
        const float b1_correction = 1.0F - std::pow(config_.beta1, static_cast<float>(step_count_));
        const float b2_correction = 1.0F - std::pow(config_.beta2, static_cast<float>(step_count_));

        for (std::size_t i = 0; i < Model::parameter_count; ++i) {
            const float g = grads[i] * gradient_scale;
            m[i] = config_.beta1 * m[i] + (1.0F - config_.beta1) * g;
            v[i] = config_.beta2 * v[i] + (1.0F - config_.beta2) * g * g;
            const float m_hat = m[i] / b1_correction;
            const float v_hat = v[i] / b2_correction;
            params[i] -= config_.learning_rate * m_hat / (std::sqrt(v_hat) + config_.epsilon);
        }
        return Status::Ok;
    }

    constexpr std::size_t step_count() const noexcept {
        return step_count_;
    }

    constexpr const Config& config() const noexcept {
        return config_;
    }

private:
    Config config_{};
    std::size_t step_count_ = 0;
};

} // namespace edge

