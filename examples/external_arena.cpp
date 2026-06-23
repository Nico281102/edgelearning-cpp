#include <array>
#include <cstddef>

#include <edge/edge.hpp>

int main() {
    using Model = edge::Model<
        edge::Backend::Generic,
        edge::Input<8>,
        edge::Dense<32, edge::ReLU>,
        edge::Dense<1>>;

    alignas(Model::alignment) static std::array<std::byte, Model::required_memory> arena{};
    Model model{edge::external_arena(arena)};
    return edge::is_ok(model.initialize(edge::InitConfig{.seed = 7U})) ? 0 : 1;
}

