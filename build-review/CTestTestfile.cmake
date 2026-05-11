# CMake generated Testfile for 
# Source directory: /Users/cat267/Documents/Projects/recombination_simulator/sparq5
# Build directory: /Users/cat267/Documents/Projects/recombination_simulator/sparq5/build-review
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(config_loader_test "/Users/cat267/Documents/Projects/recombination_simulator/sparq5/build-review/config_loader_test")
set_tests_properties(config_loader_test PROPERTIES  _BACKTRACE_TRIPLES "/Users/cat267/Documents/Projects/recombination_simulator/sparq5/CMakeLists.txt;52;add_test;/Users/cat267/Documents/Projects/recombination_simulator/sparq5/CMakeLists.txt;0;")
add_test(example_configs_smoke "/bin/sh" "/Users/cat267/Documents/Projects/recombination_simulator/sparq5/tests/run_example_configs_smoke.sh" "/Users/cat267/Documents/Projects/recombination_simulator/sparq5/build-review/sparqy" "/Users/cat267/Documents/Projects/recombination_simulator/sparq5/examples")
set_tests_properties(example_configs_smoke PROPERTIES  _BACKTRACE_TRIPLES "/Users/cat267/Documents/Projects/recombination_simulator/sparq5/CMakeLists.txt;54;add_test;/Users/cat267/Documents/Projects/recombination_simulator/sparq5/CMakeLists.txt;0;")
add_test(example_slim_smoke "/bin/sh" "/Users/cat267/Documents/Projects/recombination_simulator/sparq5/tests/run_example_slim_smoke.sh" "/usr/local/bin/slim" "/Users/cat267/Documents/Projects/recombination_simulator/sparq5/examples/slim")
set_tests_properties(example_slim_smoke PROPERTIES  _BACKTRACE_TRIPLES "/Users/cat267/Documents/Projects/recombination_simulator/sparq5/CMakeLists.txt;63;add_test;/Users/cat267/Documents/Projects/recombination_simulator/sparq5/CMakeLists.txt;0;")
add_test(slim_export_roundtrip "/bin/sh" "/Users/cat267/Documents/Projects/recombination_simulator/sparq5/tests/run_slim_export_roundtrip.sh" "/Users/cat267/Documents/Projects/recombination_simulator/sparq5/build-review/sparqy" "/usr/local/bin/slim" "/Users/cat267/Documents/Projects/recombination_simulator/sparq5/tests/slim_export_roundtrip.sparqy")
set_tests_properties(slim_export_roundtrip PROPERTIES  _BACKTRACE_TRIPLES "/Users/cat267/Documents/Projects/recombination_simulator/sparq5/CMakeLists.txt;70;add_test;/Users/cat267/Documents/Projects/recombination_simulator/sparq5/CMakeLists.txt;0;")
