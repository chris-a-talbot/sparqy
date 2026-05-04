// sparqy/src/dominance_spec.hpp
//
// Dominance models used when assigning the dominance coefficient h to each
// new mutation.

#pragma once

#include <variant>

#include "dist_spec.hpp"
#include "rng.hpp"

// Additive dominance fast-path (h = 0.5 always).
struct AdditiveDominanceSpec {
    static constexpr double additive_h = 0.5;
};

// Fixed dominance coefficient. Defaults to additive-equivalent h = 0.5.
struct FixedDominanceSpec {
    double h = 0.5;
};

// Dominance coefficient drawn from an arbitrary distribution.
// Default distribution uniform on [0, 1] (mean = 0.5 = additive).
struct DistributedDominanceSpec {
    DistSpec distribution = {DistKind::constant, 0.5, 0.0, 0.0, 1.0};
};

// Linear dominance-from-selection: h = clamp(intercept + slope * s, min_h, max_h).
// Supports the inverse h-s relationship.
struct LinearDominanceFromSelectionSpec {
    double intercept = 0.5;
    double slope     = 0.0;
    double min_h     = 0.0;
    double max_h     = 1.0;
};

// Tagged union of the four dominance models; defaults to additive.
struct DominanceSpec {
    using Model = std::variant<
        AdditiveDominanceSpec,
        FixedDominanceSpec,
        DistributedDominanceSpec,
        LinearDominanceFromSelectionSpec>;

    Model model = AdditiveDominanceSpec{};

    DominanceSpec() = default;
    DominanceSpec(const AdditiveDominanceSpec& additive) : model(additive) {}
    DominanceSpec(const FixedDominanceSpec& fixed) : model(fixed) {}
    DominanceSpec(const DistributedDominanceSpec& distributed) : model(distributed) {}
    DominanceSpec(const LinearDominanceFromSelectionSpec& linear) : model(linear) {}

    // Factories
    static DominanceSpec additive();
    static DominanceSpec fixed(double h);
    static DominanceSpec distributed(const DistSpec& distribution);
    static DominanceSpec linear_from_s(double intercept,
                                       double slope,
                                       double min_h = 0.0,
                                       double max_h = 1.0);

    bool   is_additive() const;
    double sample(RNG& rng, double s) const;
};

inline DominanceSpec DominanceSpec::additive() {
    return DominanceSpec(AdditiveDominanceSpec{});
}

inline DominanceSpec DominanceSpec::fixed(double h) {
    return DominanceSpec(FixedDominanceSpec{h});
}

inline DominanceSpec DominanceSpec::distributed(const DistSpec& distribution) {
    return DominanceSpec(DistributedDominanceSpec{distribution});
}

inline DominanceSpec DominanceSpec::linear_from_s(double intercept,
                                                  double slope,
                                                  double min_h,
                                                  double max_h) {
    return DominanceSpec(LinearDominanceFromSelectionSpec{
        intercept, slope, min_h, max_h
    });
}

inline bool DominanceSpec::is_additive() const {
    return std::holds_alternative<AdditiveDominanceSpec>(model);
}

inline double DominanceSpec::sample(RNG& rng, double s) const {
    if (std::holds_alternative<AdditiveDominanceSpec>(model)) {
        return AdditiveDominanceSpec::additive_h;
    }
    if (const auto* fixed = std::get_if<FixedDominanceSpec>(&model))
        return fixed->h;
    if (const auto* distributed = std::get_if<DistributedDominanceSpec>(&model))
        return distributed->distribution.sample(rng);

    const auto& linear = std::get<LinearDominanceFromSelectionSpec>(model);
    return clamp_value(linear.intercept + linear.slope * s,
                       linear.min_h,
                       linear.max_h);
}
