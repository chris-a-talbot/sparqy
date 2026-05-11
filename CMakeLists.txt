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

set(SPARQY_COMPILE_OPTIONS
    -O3
    -march=native
    -mtune=native
    -ffast-math
    -funroll-loops
)

function(configure_sparqy_target target_name)
    target_include_directories(${target_name} PRIVATE src)
    target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${target_name} PRIVATE ${SPARQY_COMPILE_OPTIONS})
endfunction()

add_executable(sparqy
    src/config_loader.cpp
    src/main.cpp
    src/sparqy.cpp
    src/sparqy_compile.cpp
    src/sparqy_slim_export.cpp
    src/sparqy_statistics.cpp
)
configure_sparqy_target(sparqy)

add_executable(config_loader_test
    src/config_loader.cpp
    src/sparqy.cpp
    src/sparqy_compile.cpp
    src/sparqy_slim_export.cpp
    src/sparqy_statistics.cpp
    tests/config_loader_test.cpp
)
configure_sparqy_target(config_loader_test)
add_test(NAME config_loader_test COMMAND config_loader_test)

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
    add_test(
        NAME slim_export_roundtrip
        COMMAND /bin/sh
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_slim_export_roundtrip.sh
                $<TARGET_FILE:sparqy>
                ${SLIM_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/slim_export_roundtrip.sparqy
    )
endif()
