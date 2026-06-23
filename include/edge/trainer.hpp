#pragma once

#include <array>
#include <cstddef>

#include <edge/losses.hpp>
#include <edge/model.hpp>
#include <edge/optimizers/adam.hpp>
#include <edge/optimizers/sgd.hpp>
#include <edge/status.hpp>
#include <edge/tensor.hpp>

namespace edge {

enum class GradientReduction {
    Sum,
    Mean
};

struct TrainerConfig {
    std::size_t batch_size = 1;
    GradientReduction reduction = GradientReduction::Mean;
};

template<typename ModelT, typename LossT = MSE, typename OptimizerT = SGD>
class Trainer {
public:
    using Model = ModelT;
    using Loss = LossT;
    using Optimizer = OptimizerT;
    using OptimizerConfig = typename Optimizer::Config;

    Trainer() noexcept
        : model_(&owned_model_) {}

    explicit Trainer(TrainerConfig config, OptimizerConfig optimizer_config = {}) noexcept
        : model_(&owned_model_),
          optimizer_(optimizer_config),
          config_(config) {}

    explicit Trainer(OptimizerConfig optimizer_config) noexcept
        : model_(&owned_model_),
          optimizer_(optimizer_config) {}

    explicit Trainer(Model& model,
                     TrainerConfig config = {},
                     OptimizerConfig optimizer_config = {}) noexcept
        : model_(&model),
          optimizer_(optimizer_config),
          config_(config) {}

    Status train_step(TensorView<const float, Model::input_size> input,
                      TensorView<const float, Model::output_size> target) noexcept {
        if (config_.batch_size == 0U) {
            return Status::InvalidArgument;
        }

        Status status = model_->forward(input);
        if (status != Status::Ok) {
            return status;
        }

        auto prediction = model_->output();
        TensorView<float, Model::output_size> gradient(output_gradient_);
        last_loss_ = evaluate_loss(prediction, target, gradient);

        status = model_->backward(TensorView<const float, Model::output_size>(output_gradient_));
        if (status != Status::Ok) {
            return status;
        }

        ++accumulated_samples_;
        if (accumulated_samples_ >= config_.batch_size) {
            return flush();
        }
        return Status::Ok;
    }

    template<typename InputContainer, typename TargetContainer>
    Status train_step(const InputContainer& input, const TargetContainer& target) noexcept {
        return train_step(TensorView<const float, Model::input_size>(input),
                          TensorView<const float, Model::output_size>(target));
    }

    Status flush() noexcept {
        if (accumulated_samples_ == 0U) {
            return Status::Ok;
        }
        const float scale = config_.reduction == GradientReduction::Mean
                                ? 1.0F / static_cast<float>(accumulated_samples_)
                                : 1.0F;
        Status status = optimizer_.step(*model_, scale);
        if (status != Status::Ok) {
            return status;
        }
        status = model_->zero_grad();
        if (status != Status::Ok) {
            return status;
        }
        accumulated_samples_ = 0;
        return Status::Ok;
    }

    Model& model() noexcept {
        return *model_;
    }

    const Model& model() const noexcept {
        return *model_;
    }

    Optimizer& optimizer() noexcept {
        return optimizer_;
    }

    const Optimizer& optimizer() const noexcept {
        return optimizer_;
    }

    float last_loss() const noexcept {
        return last_loss_;
    }

    std::size_t accumulated_samples() const noexcept {
        return accumulated_samples_;
    }

private:
    template<typename Prediction, typename Target, typename Gradient>
    float evaluate_loss(const Prediction& prediction,
                        const Target& target,
                        Gradient& gradient) noexcept {
        if constexpr (requires { Loss::evaluate(prediction, target, gradient); }) {
            return Loss::evaluate(prediction, target, gradient);
        } else {
            return loss_.evaluate(prediction, target, gradient);
        }
    }

    Model owned_model_{};
    Model* model_ = nullptr;
    Loss loss_{};
    Optimizer optimizer_{};
    TrainerConfig config_{};
    std::array<float, Model::output_size> output_gradient_{};
    std::size_t accumulated_samples_ = 0;
    float last_loss_ = 0.0F;
};

} // namespace edge

