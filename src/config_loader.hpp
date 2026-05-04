#pragma once

#include <string>
#include <vector>

#include "sparqy.hpp"

struct LoadedConfig {
    SimParams params;
    bool has_statistics = false;
    std::vector<std::string> warnings;
};

LoadedConfig load_config_file(const std::string& path);
