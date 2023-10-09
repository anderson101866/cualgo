cmake_minimum_required(VERSION 3.9)
project(cualgo VERSION 0.1.0)

add_subdirectory(src/cualgo)
add_subdirectory(test)

option(cualgo_build_tests "Build all of cualgo's own tests." OFF) #TODO
