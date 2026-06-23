#pragma once

#include <cstddef>

namespace edge {

struct ProfilingCounters {
    std::size_t forward_calls = 0;
    std::size_t backward_calls = 0;
    std::size_t optimizer_steps = 0;
    std::size_t zero_grad_calls = 0;
};

} // namespace edge

