cmake_minimum_required(VERSION 3.16)
project(sparqy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()
find_program(SLIM_EXECUTABLE slim)

# OpenMP — libomp on macOS / Apple Silicon lives under homebrew
if(APPLE)
    set(OpenMP_CXX_FLAGS
        "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()
find_package(OpenMP REQUIRED)

add_executable(sparqy
    src/config_loader.cpp
    src/main.cpp
    src/sparqy.cpp
    src/sparqy_compile.cpp
    src/sparqy_statistics.cpp
)
target_include_directories(sparqy PRIVATE src)
target_link_libraries(sparqy PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(sparqy PRIVATE
    -O3
    -march=native
    -mtune=native
    -ffast-math
    -funroll-loops
)

add_executable(alias_bench tests/alias_bench.cpp)
target_include_directories(alias_bench PRIVATE src)
target_link_libraries(alias_bench PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(alias_bench PRIVATE
    -O3
    -march=native
    -mtune=native
    -ffast-math
    -funroll-loops
)

add_executable(config_loader_test
    src/config_loader.cpp
    src/sparqy.cpp
    src/sparqy_compile.cpp
    src/sparqy_statistics.cpp
    tests/config_loader_test.cpp
)
target_include_directories(config_loader_test PRIVATE src)
target_link_libraries(config_loader_test PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME config_loader_test COMMAND config_loader_test)

add_executable(config_loader_bench
    src/config_loader.cpp
    tests/config_loader_bench.cpp
)
target_include_directories(config_loader_bench PRIVATE src)
target_link_libraries(config_loader_bench PRIVATE OpenMP::OpenMP_CXX)

add_test(
    NAME example_configs_smoke
    COMMAND /bin/sh
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_example_configs_smoke.sh
            $<TARGET_FILE:sparqy>
            ${CMAKE_CURRENT_SOURCE_DIR}/examples
)

if(SLIM_EXECUTABLE)
    add_test(
        NAME example_slim_smoke
        COMMAND /bin/sh
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_example_slim_smoke.sh
                ${SLIM_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/examples/slim
    )
endif()
