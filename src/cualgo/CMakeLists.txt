cmake_minimum_required(VERSION 3.9)
project(cualgo LANGUAGES CUDA CXX)

find_package(pybind11 REQUIRED)
pybind11_add_module(cualgo main.cu)

target_include_directories(cualgo PRIVATE ${CMAKE_SOURCE_DIR}/include)

#configure debug version
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if(uppercase_CMAKE_BUILD_TYPE MATCHES DEBUG)
    target_compile_options(cualgo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
endif(uppercase_CMAKE_BUILD_TYPE MATCHES DEBUG)
set_property(TARGET cualgo PROPERTY CUDA_ARCHITECTURES 80-real 86-virtual)

#make host/device codes relocatable
set_target_properties(cualgo PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(cualgo
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # -dc
if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET cualgo 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()