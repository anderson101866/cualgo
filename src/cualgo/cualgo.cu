//This file defines the entry point for a python module
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cualgo_backends/graph/floydwarshall.cuh"

namespace bk = cualgo_backends;
namespace py = pybind11;

static inline constexpr auto kDocFloydWarshall = "Apply Floyd-Warshall algorithm to find shortest path of each pair of source/sink.";

PYBIND11_MODULE(cualgo, m) {
    m.doc() = R"(CuAlgo
=====
A Pytnon library containing basic algorithm with GPU-accelerated computing.

Provide the following facility with CUDA implementation:
  Graph-Related Algorithm
  - Floyd-Warshall
)";

    m.def("floydwarshall", 
          py::overload_cast<std::vector<std::vector<double>>&&>(&bk::graph::FloydWarshallDriver<double>), 
          kDocFloydWarshall);
    m.def("floydwarshall", 
          py::overload_cast<std::vector<std::vector<int>>&&>(&bk::graph::FloydWarshallDriver<int>), 
          kDocFloydWarshall);
}