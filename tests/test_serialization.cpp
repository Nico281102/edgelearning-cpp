#include <array>

#include <edge/edge.hpp>

#include "test_harness.hpp"

int main() {
    using Model = edge::Model<edge::InputVector<2>, edge::Dense<2>, edge::Dense<1>>;
    Model a;
    Model b;
    std::array<float, Model::parameter_count> params{};
    for (std::size_t i = 0; i < params.size(); ++i) {
        params[i] = static_cast<float>(i) * 0.125F;
    }

    EDGE_EXPECT_EQ(a.import_parameters(params), edge::Status::Ok);
    std::array<float, Model::parameter_count> exported{};
    EDGE_EXPECT_EQ(a.export_parameters(exported), edge::Status::Ok);
    EDGE_EXPECT_EQ(b.import_parameters(exported), edge::Status::Ok);

    for (std::size_t i = 0; i < Model::parameter_count; ++i) {
        EDGE_EXPECT_NEAR(b.parameter_data()[i], params[i], 1.0e-7F);
    }

    EDGE_EXPECT_EQ(a.export_parameters(nullptr, Model::parameter_count), edge::Status::NullPointer);
    float too_small[1]{};
    EDGE_EXPECT_EQ(a.export_parameters(too_small, 1), edge::Status::InvalidBufferLength);
}

