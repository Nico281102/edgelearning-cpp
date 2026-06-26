#include <array>
#include <type_traits>

#include <edge/edge.hpp>

struct CustomFallbackBackend {
    static constexpr bool is_backend_policy = true;
};

struct CustomNoFallbackBackend {
    static constexpr bool is_backend_policy = true;
    static constexpr bool falls_back_to_generic = false;
};

int main() {
    using DefaultBackendModel = edge::Model<
        edge::InputVector<8>,
        edge::Dense<32, edge::ReLU>,
        edge::Dense<16, edge::ReLU>,
        edge::Dense<1>>;

    using ExplicitBackendModel = edge::Model<
        edge::Backend::Generic,
        edge::InputVector<8>,
        edge::Dense<32, edge::ReLU>,
        edge::Dense<1>>;

    using M55BackendModel = edge::Model<
        edge::Backend::M55,
        edge::InputVector<32>,
        edge::Dense<8, edge::ReLU>,
        edge::Dense<8, edge::ReLU>,
        edge::Dense<1>>;
    using ConvShapeModel = edge::Model<
        edge::Input<edge::CHW<1, 3, 3>>,
        edge::Conv2D<2, edge::Kernel<2, 2>>,
        edge::Flatten,
        edge::Dense<1>>;
    using CustomBackendModel = edge::Model<
        CustomFallbackBackend,
        edge::InputVector<2>,
        edge::Dense<1>>;

    static_assert(edge::Vector<8>::layout == edge::Layout::Flat);
    static_assert(edge::Vector<8>::elements == 8);
    static_assert(edge::CHW<1, 3, 3>::layout == edge::Layout::CHW);
    static_assert(edge::CHW<1, 3, 3>::elements == 9);
    static_assert(edge::shape_dim_v<1, typename edge::CHW<1, 3, 3>::shape> == 3);
    static_assert(DefaultBackendModel::input_size == 8);
    static_assert(DefaultBackendModel::output_size == 1);
    static_assert(
        std::is_same_v<typename DefaultBackendModel::input_spec, edge::Vector<8>>);
    static_assert(
        std::is_same_v<typename DefaultBackendModel::output_spec, edge::Vector<1>>);
    static_assert(DefaultBackendModel::layer_count == 3);
    static_assert(DefaultBackendModel::parameter_count == (8 * 32 + 32 + 32 * 16 + 16 + 16 * 1 + 1));
    static_assert(DefaultBackendModel::required_memory >= DefaultBackendModel::total_bytes);
    static_assert(!std::is_copy_constructible_v<DefaultBackendModel>);
    static_assert(!std::is_copy_assignable_v<DefaultBackendModel>);
    static_assert(!std::is_move_constructible_v<DefaultBackendModel>);
    static_assert(!std::is_move_assignable_v<DefaultBackendModel>);
    static_assert(std::is_same_v<typename ExplicitBackendModel::backend, edge::Backend::Generic>);
    static_assert(M55BackendModel::alignment >= 32);
    static_assert((M55BackendModel::parameter_offset % 32) == 0);
    static_assert((M55BackendModel::gradient_offset % 32) == 0);
    static_assert((M55BackendModel::optimizer_offset % 32) == 0);
    static_assert((M55BackendModel::activation_offset % 32) == 0);
    static_assert((M55BackendModel::workspace_offset % 32) == 0);
    static_assert(edge::backend_falls_back_to_generic_v<edge::Backend::M55>);
    static_assert(edge::BackendPolicy<CustomFallbackBackend>);
    static_assert(edge::BackendPolicy<CustomNoFallbackBackend>);
    static_assert(edge::backend_falls_back_to_generic_v<CustomFallbackBackend>);
    static_assert(!edge::backend_falls_back_to_generic_v<CustomNoFallbackBackend>);
    static_assert(ConvShapeModel::input_size == 9);
    static_assert(ConvShapeModel::output_size == 1);
    static_assert(ConvShapeModel::parameter_count == (2 * 1 * 2 * 2 + 2 + 8 * 1 + 1));
    static_assert(std::is_same_v<typename CustomBackendModel::backend, CustomFallbackBackend>);

    alignas(DefaultBackendModel::alignment)
        std::array<std::byte, DefaultBackendModel::required_memory> arena{};
    auto external = edge::external_arena(arena);
    static_assert(decltype(external)::size == DefaultBackendModel::required_memory);
}
