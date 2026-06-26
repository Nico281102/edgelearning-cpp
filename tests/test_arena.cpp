#include <array>
#include <cstddef>
#include <span>

#include <edge/edge.hpp>

#include "test_harness.hpp"

int main() {
    using Model = edge::Model<
        edge::Backend::Generic,
        edge::InputVector<3>,
        edge::Dense<4, edge::ReLU>,
        edge::Dense<1>>;

    static_assert(Model::required_memory > 0);
    static_assert(Model::parameter_bytes > 0);
    static_assert(Model::gradient_bytes == Model::parameter_bytes);

    Model owned;
    EDGE_EXPECT_EQ(owned.status(), edge::Status::Ok);

    alignas(Model::alignment) std::array<std::byte, Model::required_memory> arena{};
    Model external{edge::external_arena(arena)};
    EDGE_EXPECT_EQ(external.status(), edge::Status::Ok);

    alignas(Model::alignment) std::byte c_arena[Model::required_memory]{};
    Model external_c{edge::external_arena(c_arena)};
    EDGE_EXPECT_EQ(external_c.status(), edge::Status::Ok);

    std::span<std::byte, Model::required_memory> span_arena(arena);
    Model external_span{edge::external_arena(span_arena)};
    EDGE_EXPECT_EQ(external_span.status(), edge::Status::Ok);
}

