// sparqy/src/dist_spec.hpp
//
// Parametric probability distributions used for DFE and dominance sampling.

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#include "rng.hpp"

// Clamp a value x to [lo, hi]. Shared between DistSpec::sample and
// DominanceSpec::sample; defined here because dist_spec.hpp is the earliest
// header in the sampling chain.
inline double clamp_value(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// Type of probability distribution
enum class DistKind : uint8_t {
    constant, uniform, normal, exponential, gamma, beta
};

// Specification for a parametric distribution.
// kind = type; p1 and p2 = parameters; min_value / max_value = post-sample clamp
struct DistSpec {
    DistKind kind = DistKind::constant;

    double p1 = 0.0;
    double p2 = 0.0;

    double min_value = -std::numeric_limits<double>::max();
    double max_value =  std::numeric_limits<double>::max();

    double sample(RNG& rng) const;
};

inline double DistSpec::sample(RNG& rng) const {
    double x = p1;
    switch (kind) {
        case DistKind::constant:    x = p1; break;
        case DistKind::uniform:     x = p1 + (p2 - p1) * rng.uniform(); break;
        case DistKind::normal:      x = p1 + p2 * rng.normal(); break;
        case DistKind::exponential: x = rng.exponential(p1); break;
        case DistKind::gamma: {
            if (p2 <= 0.0 || p1 == 0.0) {
                x = 0.0;
            } else {
                double scale = std::abs(p1) / p2;
                double draw  = rng.gamma(p2, scale);
                x = (p1 < 0.0) ? -draw : draw;
            }
            break;
        }
        case DistKind::beta:
            x = rng.beta(p1, p2);
            break;
    }
    return clamp_value(x, min_value, max_value);
}
