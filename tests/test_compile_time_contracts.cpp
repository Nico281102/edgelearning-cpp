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

    alignas(DefaultBackendModel::alignment)
        std::array<std::byte, DefaultBackendModel::required_memory> arena{};
    auto external = edge::external_arena(arena);
    static_assert(decltype(external)::size == DefaultBackendModel::required_memory);
}

