#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

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

    template<typename Model, typename ScaleT>
    Status step(Model& model, ScaleT gradient_scale) noexcept {
        using ParameterT = typename Model::parameter_type;
        using GradientT = typename Model::gradient_type;
        using OptimizerStateT = typename Model::optimizer_state_type;
        using ComputeT = std::common_type_t<typename Model::accumulator_type, ScaleT, float>;
        ParameterT* params = model.parameter_data();
        const GradientT* grads = model.gradient_data();
        OptimizerStateT* state = model.optimizer_state_data();
        if (params == nullptr || grads == nullptr || state == nullptr) {
            return Status::NullPointer;
        }

        ++step_count_;
        OptimizerStateT* m = state;
        OptimizerStateT* v = state + Model::parameter_count;
        const ComputeT beta1 = static_cast<ComputeT>(config_.beta1);
        const ComputeT beta2 = static_cast<ComputeT>(config_.beta2);
        const ComputeT b1_correction =
            ComputeT{1} - static_cast<ComputeT>(std::pow(beta1, static_cast<ComputeT>(step_count_)));
        const ComputeT b2_correction =
            ComputeT{1} - static_cast<ComputeT>(std::pow(beta2, static_cast<ComputeT>(step_count_)));

        for (std::size_t i = 0; i < Model::parameter_count; ++i) {
            const ComputeT g =
                static_cast<ComputeT>(grads[i]) * static_cast<ComputeT>(gradient_scale);
            const ComputeT next_m =
                beta1 * static_cast<ComputeT>(m[i]) + (ComputeT{1} - beta1) * g;
            const ComputeT next_v =
                beta2 * static_cast<ComputeT>(v[i]) + (ComputeT{1} - beta2) * g * g;
            m[i] = static_cast<OptimizerStateT>(next_m);
            v[i] = static_cast<OptimizerStateT>(next_v);
            const ComputeT m_hat = next_m / b1_correction;
            const ComputeT v_hat = next_v / b2_correction;
            const ComputeT update = static_cast<ComputeT>(config_.learning_rate) * m_hat /
                                    (static_cast<ComputeT>(std::sqrt(v_hat)) +
                                     static_cast<ComputeT>(config_.epsilon));
            params[i] = static_cast<ParameterT>(static_cast<ComputeT>(params[i]) - update);
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
