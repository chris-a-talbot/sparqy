// sparqy/src/rng.hpp — xoshiro256** random number generator
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

struct RNG {
    uint64_t s[4] = {};

    void seed(uint64_t v) noexcept {
        // SplitMix64 to initialize state from a single seed
        for (int i = 0; i < 4; i++) {
            uint64_t z = (v += 0x9e3779b97f4a7c15ULL);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }

    uint64_t next_raw() noexcept {
        const uint64_t res = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return res;
    }

    // Advance state by 2^192 draws. Two RNGs separated by a long_jump are
    // guaranteed to produce non-overlapping sequences for 2^64 calls, which
    // is enough for any simulation on any conceivable hardware.
    void long_jump() noexcept {
        static constexpr uint64_t JUMP[] = {
            0x76e15d3efefdcbbfULL, 0xc5004e441c522fb3ULL,
            0x77710069854ee241ULL, 0x39109bb02acbe635ULL
        };
        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < 4; i++) {
            for (int b = 0; b < 64; b++) {
                if (JUMP[i] & (1ULL << b)) {
                    s0 ^= s[0]; s1 ^= s[1]; s2 ^= s[2]; s3 ^= s[3];
                }
                next_raw();
            }
        }
        s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
    }

    // Uniform double in [0, 1)
    double uniform() noexcept {
        return (next_raw() >> 11) * (1.0 / (1ULL << 53));
    }

    // Uniform integer in [0, n)
    uint64_t uniform_int(uint64_t n) noexcept {
        __uint128_t m = (__uint128_t)next_raw() * n;
        uint64_t l = (uint64_t)m;
        if (l < n) {
            uint64_t t = (-n) % n;
            while (l < t) { m = (__uint128_t)next_raw() * n; l = (uint64_t)m; }
        }
        return (uint64_t)(m >> 64);
    }

    // Standard normal via Box-Muller
    double normal() noexcept {
        double u1 = std::max(1e-300, uniform());
        double u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }

    // Exponential with given mean
    double exponential(double mean) noexcept {
        return -mean * std::log(std::max(1e-300, uniform()));
    }

    // Gamma via Marsaglia-Tsang
    double gamma(double shape, double scale) noexcept {
        if (shape < 1.0)
            return gamma(shape + 1.0, scale) * std::pow(std::max(1e-300, uniform()), 1.0 / shape);
        double d = shape - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
        for (;;) {
            double x = normal(), v = 1.0 + c * x;
            if (v <= 0.0) continue;
            v = v * v * v;
            double u = uniform();
            if (u < 1.0 - 0.0331 * x * x * x * x) return scale * d * v;
            if (std::log(std::max(1e-300, u)) < 0.5 * x * x + d * (1.0 - v + std::log(v)))
                return scale * d * v;
        }
    }

    // Beta(a, b)
    double beta(double a, double b) noexcept {
        double x = gamma(a, 1.0), y = gamma(b, 1.0);
        return (x + y > 0.0) ? x / (x + y) : 0.5;
    }

    // Poisson
    uint32_t poisson(double lambda) noexcept {
        if (lambda <= 0.0) return 0;
        if (lambda < 30.0) {
            double L = std::exp(-lambda), p = 1.0;
            uint32_t k = 0;
            do { ++k; p *= uniform(); } while (p > L);
            return k - 1;
        }
        double n = std::sqrt(-2.0 * std::log(std::max(1e-300, uniform())))
                   * std::cos(6.283185307179586 * uniform());
        return (uint32_t)std::max(0, (int)std::llround(lambda + n * std::sqrt(lambda)));
    }

    // Fast path: caller has precomputed exp_neg_lambda = exp(-lambda).
    // Valid only when lambda < 30 and > 0.
    uint32_t poisson_precomputed(double exp_neg_lambda) noexcept {
        double p = 1.0;
        uint32_t k = 0;
        do { ++k; p *= uniform(); } while (p > exp_neg_lambda);
        return k - 1;
    }

private:
    static uint64_t rotl(uint64_t x, int k) noexcept { return (x << k) | (x >> (64 - k)); }
};
