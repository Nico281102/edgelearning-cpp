#include <edge/backend.hpp>

// M55-specific Dense kernels live in include/edge/backends/m55.hpp as template
// hooks so unsupported host builds can fall back without linking target objects.
// Legacy C backends are consumed only by optional test targets from an external
// checkout; their sources are not vendored here.
