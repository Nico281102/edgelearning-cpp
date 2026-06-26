#include <array>
#include <type_traits>

#include <edge/edge.hpp>

int main() {
    using DefaultBackendModel = edge::Model<
        edge::Input<8>,
        edge::Dense<32, edge::ReLU>,
        edge::Dense<16, edge::ReLU>,
        edge::Dense<1>>;

    using ExplicitBackendModel = edge::Model<
        edge::Backend::Generic,
        edge::Input<8>,
        edge::Dense<32, edge::ReLU>,
        edge::Dense<1>>;

    using M55BackendModel = edge::Model<
        edge::Backend::M55,
        edge::Input<32>,
        edge::Dense<8, edge::ReLU>,
        edge::Dense<8, edge::ReLU>,
        edge::Dense<1>>;

    static_assert(DefaultBackendModel::input_size == 8);
    static_assert(DefaultBackendModel::output_size == 1);
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

    alignas(DefaultBackendModel::alignment)
        std::array<std::byte, DefaultBackendModel::required_memory> arena{};
    auto external = edge::external_arena(arena);
    static_assert(decltype(external)::size == DefaultBackendModel::required_memory);
}
