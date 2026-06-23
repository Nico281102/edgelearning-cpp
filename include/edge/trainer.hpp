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
    using activation_type = typename Model::activation_type;
    using accumulator_type = typename Model::accumulator_type;
    using loss_type = typename Model::loss_type;

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

    Status train_step(TensorView<const activation_type, Model::input_size> input,
                      TensorView<const activation_type, Model::output_size> target) noexcept {
        if (config_.batch_size == 0U) {
            return Status::InvalidArgument;
        }

        Status status = model_->forward(input);
        if (status != Status::Ok) {
            return status;
        }

        auto prediction = model_->output();
        TensorView<accumulator_type, Model::output_size> gradient(output_gradient_);
        last_loss_ = evaluate_loss(prediction, target, gradient);

        status = model_->backward(
            TensorView<const accumulator_type, Model::output_size>(output_gradient_));
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
        return train_step(TensorView<const activation_type, Model::input_size>(input),
                          TensorView<const activation_type, Model::output_size>(target));
    }

    Status flush() noexcept {
        if (accumulated_samples_ == 0U) {
            return Status::Ok;
        }
        const accumulator_type scale =
            config_.reduction == GradientReduction::Mean
                ? accumulator_type{1} / static_cast<accumulator_type>(accumulated_samples_)
                : accumulator_type{1};
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

    loss_type last_loss() const noexcept {
        return last_loss_;
    }

    std::size_t accumulated_samples() const noexcept {
        return accumulated_samples_;
    }

private:
    template<typename Prediction, typename Target, typename Gradient>
    loss_type evaluate_loss(const Prediction& prediction,
                            const Target& target,
                            Gradient& gradient) noexcept {
        if constexpr (requires { Loss::evaluate(prediction, target, gradient); }) {
            return static_cast<loss_type>(Loss::evaluate(prediction, target, gradient));
        } else {
            return static_cast<loss_type>(loss_.evaluate(prediction, target, gradient));
        }
    }

    Model owned_model_{};
    Model* model_ = nullptr;
    Loss loss_{};
    Optimizer optimizer_{};
    TrainerConfig config_{};
    std::array<accumulator_type, Model::output_size> output_gradient_{};
    std::size_t accumulated_samples_ = 0;
    loss_type last_loss_{};
};

} // namespace edge
