cmake_minimum_required(VERSION 3.9)
project(cualgo)

add_subdirectory(src/cualgo)
add_subdirectory(tests)

option(cualgo_build_tests "Build all of cualgo's own tests." OFF) #TODO: C++ test?

target_compile_definitions(cualgo
                           PRIVATE VERSION_INFO=${CUALGO_VERSION_INFO})
