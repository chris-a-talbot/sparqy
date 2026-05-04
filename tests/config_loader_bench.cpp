#include "config_loader.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            std::fprintf(stderr,
                         "usage: config_loader_bench CONFIG_PATH [REPETITIONS]\n");
            return 1;
        }

        const std::string config_path = argv[1];
        const int repetitions = (argc == 3) ? std::atoi(argv[2]) : 1000;
        if (repetitions <= 0) {
            std::fprintf(stderr, "repetitions must be positive\n");
            return 1;
        }

        volatile uint64_t guard = 0u;
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < repetitions; ++i) {
            const LoadedConfig loaded = load_config_file(config_path);
            guard += (uint64_t)loaded.params.N;
            guard += (uint64_t)loaded.params.G;
            guard += (uint64_t)loaded.params.chromosomes.size();
            guard += (uint64_t)loaded.params.mutation_types.size();
            guard += (uint64_t)loaded.params.statistic_requests.size();
        }
        const auto end = std::chrono::steady_clock::now();
        const double total_ms =
            std::chrono::duration<double, std::milli>(end - begin).count();
        const double avg_ms = total_ms / (double)repetitions;

        std::printf("config=%s repetitions=%d total_ms=%.6f avg_ms=%.6f guard=%llu\n",
                    config_path.c_str(),
                    repetitions,
                    total_ms,
                    avg_ms,
                    (unsigned long long)guard);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "config_loader_bench failed: %s\n", e.what());
        return 1;
    }

    return 0;
}
