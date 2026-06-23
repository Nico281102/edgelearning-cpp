#pragma once

#include <cstddef>

#include <edge/status.hpp>

namespace edge {

struct SGDConfig {
    float learning_rate = 0.01F;
};

class SGD {
public:
    using Config = SGDConfig;

    explicit constexpr SGD(Config config = {}) noexcept
        : config_(config) {}

    template<typename Model>
    Status step(Model& model, float gradient_scale) noexcept {
        float* params = model.parameter_data();
        const float* grads = model.gradient_data();
        if (params == nullptr || grads == nullptr) {
            return Status::NullPointer;
        }
        for (std::size_t i = 0; i < Model::parameter_count; ++i) {
            params[i] -= config_.learning_rate * grads[i] * gradient_scale;
        }
        ++step_count_;
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

