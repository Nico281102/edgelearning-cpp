#include <array>

#include <edge/edge.hpp>

int main() {
    using GenericModel = edge::Model<
        edge::Backend::Generic,
        edge::InputVector<4>,
        edge::Dense<8, edge::ReLU>,
        edge::Dense<1>>;

    using M55Model = edge::Model<
        edge::Backend::M55,
        edge::InputVector<4>,
        edge::Dense<8, edge::ReLU>,
        edge::Dense<1>>;

    static_assert(GenericModel::parameter_count == M55Model::parameter_count);
    static_assert(GenericModel::output_size == M55Model::output_size);

    M55Model model;
    model.initialize(edge::InitConfig{.seed = 11U});

    const std::array<float, M55Model::input_size> input{0.0F, 0.25F, -0.5F, 1.0F};
    return edge::is_ok(model.forward(input)) ? 0 : 1;
}
