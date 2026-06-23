#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define EDGE_EXPECT_TRUE(expr)                                                       \
    do {                                                                             \
        if (!(expr)) {                                                               \
            std::fprintf(stderr, "EXPECT_TRUE failed: %s at %s:%d\n", #expr,        \
                         __FILE__, __LINE__);                                        \
            std::abort();                                                            \
        }                                                                            \
    } while (false)

#define EDGE_EXPECT_EQ(a, b) EDGE_EXPECT_TRUE((a) == (b))

inline void edge_expect_near(float actual,
                             float expected,
                             float tolerance,
                             const char* expr,
                             const char* file,
                             int line) {
    if (std::fabs(actual - expected) > tolerance) {
        std::fprintf(stderr,
                     "EXPECT_NEAR failed: %s actual=%g expected=%g tol=%g at %s:%d\n",
                     expr,
                     static_cast<double>(actual),
                     static_cast<double>(expected),
                     static_cast<double>(tolerance),
                     file,
                     line);
        std::abort();
    }
}

#define EDGE_EXPECT_NEAR(actual, expected, tolerance)                                \
    edge_expect_near((actual), (expected), (tolerance), #actual, __FILE__, __LINE__)

