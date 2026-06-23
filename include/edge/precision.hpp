#pragma once

namespace edge {

struct UseModelPrecision {};

namespace precision {

struct FP32 {
    using ParameterT = float;
    using ActivationT = float;
    using GradientT = float;
    using AccumulatorT = float;
    using OptimizerStateT = float;
    using LossT = float;
};

struct MixedFP16 {
    using ParameterT = float;
    using ActivationT = float;
    using GradientT = float;
    using AccumulatorT = float;
    using OptimizerStateT = float;
    using LossT = float;
};

} // namespace precision

} // namespace edge

