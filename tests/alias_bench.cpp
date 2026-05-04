// alias_bench.cpp — microbench + correctness for AliasSampler::build_parallel
// (PSA) and AliasSampler::build_parallel_psa_plus (PSA+).
//
// Drives both implementations on the same input weights, samples a large
// number of draws, and reports max chi-squared-style deviation per bin plus
// build timing. Used to A/B PSA vs PSA+.

#include "../src/alias_sampler.hpp"
#include "../src/rng.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <omp.h>

using clk = std::chrono::steady_clock;

static std::vector<double> gen_uniform(int n, uint64_t seed) {
    std::vector<double> w(n);
    RNG rng; rng.seed(seed);
    for (int i = 0; i < n; ++i) w[i] = rng.uniform();
    return w;
}

static std::vector<double> gen_exponential(int n, uint64_t seed) {
    std::vector<double> w(n);
    RNG rng; rng.seed(seed);
    for (int i = 0; i < n; ++i) w[i] = rng.exponential(1.0);
    return w;
}

static std::vector<double> gen_powerlaw(int n, double s, uint64_t seed) {
    std::vector<double> w(n);
    for (int i = 0; i < n; ++i) w[i] = std::pow((double)(i + 1), -s);
    RNG rng; rng.seed(seed);
    for (int i = n - 1; i > 0; --i) {
        uint64_t j = rng.uniform_int((uint64_t)(i + 1));
        std::swap(w[i], w[(int)j]);
    }
    return w;
}

// Approximates sparqy's individual-fitness distribution: products of (1 - h*s)
// terms with s ~ U(0, S_max), Poisson-distributed mutation count. Heavy-tailed
// but mostly clustered near 1.
static std::vector<double> gen_sparqy_like(int n, double mean_muts,
                                           double s_max, uint64_t seed) {
    std::vector<double> w(n);
    RNG rng; rng.seed(seed);
    for (int i = 0; i < n; ++i) {
        double f = 1.0;
        uint32_t m = rng.poisson(mean_muts);
        for (uint32_t k = 0; k < m; ++k) f *= 1.0 - s_max * rng.uniform();
        w[i] = std::max(f, 0.0);
    }
    return w;
}

// One pathological case: nearly-uniform weights with one giant outlier.
// Many lights, very few heavies — local greedy will starve for heavies in
// most chunks.
static std::vector<double> gen_one_giant(int n) {
    std::vector<double> w(n, 1.0);
    if (n > 0) w[n / 2] = 1e6;
    return w;
}

// Symmetric pathological case: nearly-uniform but one tiny outlier.
static std::vector<double> gen_one_tiny(int n) {
    std::vector<double> w(n, 1.0);
    if (n > 0) w[n / 2] = 1e-6;
    return w;
}

struct VerifyResult {
    double max_dev_sigma;
    int    bins_checked;
};

static VerifyResult verify_sampler(AliasSampler& s, const std::vector<double>& w,
                                   uint64_t sample_seed, int draws_per_item) {
    const int n = (int)w.size();
    std::vector<uint64_t> hits(n, 0);
    RNG rng; rng.seed(sample_seed);
    const int64_t total = (int64_t)n * draws_per_item;
    for (int64_t i = 0; i < total; ++i) {
        int x = s.sample(rng);
        if (x < 0 || x >= n) {
            std::fprintf(stderr, "FATAL: sampler returned out-of-range index %d\n", x);
            std::exit(2);
        }
        ++hits[x];
    }
    double sumw = 0.0;
    for (double x : w) sumw += x;
    double max_dev = 0.0;
    int bins_checked = 0;
    for (int i = 0; i < n; ++i) {
        double expected = (double)total * (w[i] / sumw);
        if (expected < 5.0) continue; // skip thin bins (chi-squared assumption)
        double observed = (double)hits[i];
        double sigma = std::abs(observed - expected) / std::sqrt(expected);
        if (sigma > max_dev) max_dev = sigma;
        ++bins_checked;
    }
    return {max_dev, bins_checked};
}

struct TimingStats {
    double min_ms, med_ms, max_ms;
};

static TimingStats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v.front(), v[v.size() / 2], v.back()};
}

int main(int argc, char** argv) {
    int n = 1 << 20;
    int threads = 0;
    int reps = 7;
    int draws_per_item = 30;
    bool run_psa = true;
    bool run_psa_plus = true;
    std::string dist = "uniform";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto pref = [&](const char* p, std::string& out) {
            size_t L = std::strlen(p);
            if (a.size() > L && std::memcmp(a.data(), p, L) == 0) {
                out = a.substr(L); return true;
            }
            return false;
        };
        std::string v;
        if      (pref("--n=", v))         n = std::atoi(v.c_str());
        else if (pref("--threads=", v))   threads = std::atoi(v.c_str());
        else if (pref("--reps=", v))      reps = std::atoi(v.c_str());
        else if (pref("--draws=", v))     draws_per_item = std::atoi(v.c_str());
        else if (pref("--dist=", v))      dist = v;
        else if (a == "--psa-only")       run_psa_plus = false;
        else if (a == "--psa-plus-only")  run_psa = false;
        else if (a == "-h" || a == "--help") {
            std::printf("usage: alias_bench [--n=N] [--threads=T] [--reps=R] "
                        "[--draws=D] [--dist=NAME] [--psa-only|--psa-plus-only]\n"
                        "  dists: uniform, exponential, powerlaw1, sparqy, "
                        "one_giant, one_tiny\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 1;
        }
    }
    if (threads <= 0) threads = omp_get_max_threads();

    std::vector<double> w;
    if      (dist == "uniform")     w = gen_uniform(n, 12345);
    else if (dist == "exponential") w = gen_exponential(n, 12345);
    else if (dist == "powerlaw1")   w = gen_powerlaw(n, 1.0, 12345);
    else if (dist == "sparqy")      w = gen_sparqy_like(n, 10.0, 0.01, 12345);
    else if (dist == "one_giant")   w = gen_one_giant(n);
    else if (dist == "one_tiny")    w = gen_one_tiny(n);
    else { std::fprintf(stderr, "unknown dist: %s\n", dist.c_str()); return 1; }

    std::printf("dist=%s n=%d threads=%d reps=%d draws_per_item=%d\n",
                dist.c_str(), n, threads, reps, draws_per_item);

    auto bench = [&](const char* name,
                     std::function<void(AliasSampler&)> build_fn) {
        std::vector<double> times;
        AliasSampler last;
        // warm-up
        { AliasSampler s; build_fn(s); }
        for (int r = 0; r < reps; ++r) {
            AliasSampler s;
            auto t0 = clk::now();
            build_fn(s);
            auto t1 = clk::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            if (r == reps - 1) last = std::move(s);
        }
        auto v = verify_sampler(last, w, 0xC0FFEEull, draws_per_item);
        auto t = summarize(std::move(times));
        std::printf("%-12s build min=%7.3f ms  med=%7.3f ms  max=%7.3f ms  | "
                    "verify max-dev=%.2fσ over %d bins\n",
                    name, t.min_ms, t.med_ms, t.max_ms,
                    v.max_dev_sigma, v.bins_checked);
        return t.min_ms;
    };

    double psa_min = 0.0, psaplus_min = 0.0;
    if (run_psa)      psa_min     = bench("PSA",  [&](AliasSampler& s){ s.build_parallel(w, threads); });
    if (run_psa_plus) psaplus_min = bench("PSA+", [&](AliasSampler& s){ s.build_parallel_psa_plus(w, threads); });
    if (run_psa && run_psa_plus) {
        std::printf("speedup PSA/PSA+ = %.2fx\n", psa_min / psaplus_min);
    }
    return 0;
}
