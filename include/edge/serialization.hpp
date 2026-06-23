#pragma once

namespace edge {

struct FlatParameterLayout {
    static constexpr const char* description =
        "Layer order, row-major weights [out_features x in_features], then bias.";
};

} // namespace edge

