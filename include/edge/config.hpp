#pragma once

#include <cassert>

#ifndef EDGE_ASSERT
#define EDGE_ASSERT(expr) assert(expr)
#endif

namespace edge {

inline constexpr bool backend_strict_mode =
#ifdef EDGE_BACKEND_STRICT
    true;
#else
    false;
#endif

} // namespace edge

