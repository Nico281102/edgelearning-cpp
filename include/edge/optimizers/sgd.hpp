#pragma once

#include <cstddef>
#include <type_traits>

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

    template<typename Model, typename ScaleT>
    Status step(Model& model, ScaleT gradient_scale) noexcept {
        using ParameterT = typename Model::parameter_type;
        using GradientT = typename Model::gradient_type;
        using ComputeT = std::common_type_t<typename Model::accumulator_type, ScaleT, float>;
        ParameterT* params = model.parameter_data();
        const GradientT* grads = model.gradient_data();
        if (params == nullptr || grads == nullptr) {
            return Status::NullPointer;
        }
        for (std::size_t i = 0; i < Model::parameter_count; ++i) {
            const ComputeT update = static_cast<ComputeT>(config_.learning_rate) *
                                    static_cast<ComputeT>(grads[i]) *
                                    static_cast<ComputeT>(gradient_scale);
            params[i] = static_cast<ParameterT>(static_cast<ComputeT>(params[i]) - update);
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
