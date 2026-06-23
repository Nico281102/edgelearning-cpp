#include <array>
#include <cstddef>

#include <edge/edge.hpp>

#include "test_harness.hpp"

namespace {

template<typename Model>
void load_params(Model& model) {
    for (std::size_t i = 0; i < Model::parameter_count; ++i) {
        model.parameter_data()[i] =
            0.02F * static_cast<float>((i % 11U) + 1U) - 0.12F;
    }
}

} // namespace

int main() {
    using Generic = edge::Model<
        edge::Backend::Generic,
        edge::Input<4>,
        edge::Dense<3, edge::ReLU>,
        edge::Dense<2, edge::Linear>>;
    using M55 = edge::Model<
        edge::Backend::M55,
        edge::Input<4>,
        edge::Dense<3, edge::ReLU>,
        edge::Dense<2, edge::Linear>>;

    Generic generic;
    M55 m55;
    load_params(generic);
    load_params(m55);

    const std::array<float, 4> input{0.25F, -0.50F, 0.75F, 0.10F};
    const std::array<float, 2> upstream{0.40F, -0.30F};

    EDGE_EXPECT_EQ(generic.forward(input), edge::Status::Ok);
    EDGE_EXPECT_EQ(m55.forward(input), edge::Status::Ok);
    for (std::size_t i = 0; i < Generic::output_size; ++i) {
        EDGE_EXPECT_NEAR(generic.output()[i], m55.output()[i], 1.0e-6F);
    }

    EDGE_EXPECT_EQ(generic.zero_grad(), edge::Status::Ok);
    EDGE_EXPECT_EQ(m55.zero_grad(), edge::Status::Ok);
    EDGE_EXPECT_EQ(generic.backward(upstream), edge::Status::Ok);
    EDGE_EXPECT_EQ(m55.backward(upstream), edge::Status::Ok);
    for (std::size_t i = 0; i < Generic::parameter_count; ++i) {
        EDGE_EXPECT_NEAR(generic.gradient_data()[i], m55.gradient_data()[i], 1.0e-6F);
    }

    using ConvGeneric = edge::Model<
        edge::Backend::Generic,
        edge::Input<9>,
        edge::Conv2D<1, 3, 3, 2, 2, 2, edge::ReLU>,
        edge::Dense<1, edge::Linear>>;
    using ConvM55 = edge::Model<
        edge::Backend::M55,
        edge::Input<9>,
        edge::Conv2D<1, 3, 3, 2, 2, 2, edge::ReLU>,
        edge::Dense<1, edge::Linear>>;

    ConvGeneric conv_generic;
    ConvM55 conv_m55;
    load_params(conv_generic);
    load_params(conv_m55);

    const std::array<float, 9> conv_input{
        0.1F, 0.2F, 0.3F,
        0.4F, 0.5F, 0.6F,
        0.7F, 0.8F, 0.9F};
    EDGE_EXPECT_EQ(conv_generic.forward(conv_input), edge::Status::Ok);
    EDGE_EXPECT_EQ(conv_m55.forward(conv_input), edge::Status::Ok);
    EDGE_EXPECT_NEAR(conv_generic.output()[0], conv_m55.output()[0], 1.0e-6F);
}
