#pragma once
#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "cualgo_backends/common.hpp"
#include "cualgo_backends/malloc_helper.cuh"

namespace cualgo_backends {
namespace graph {

template <typename T, int kUnrollX, int kUnrollY>
__global__ void FloydWarshallCoreInnerLoop(T *graph, const int N, const size_t kPitch, const int k) {
    static_assert(kUnrollX > 0 && kUnrollY > 0, "`kUnrollX` and `kUnrollY` should be both positive.");
    const int j = blockDim.x*kUnrollX * blockIdx.x + threadIdx.x; //coalesced memory access
    const int i = blockDim.y*kUnrollY * blockIdx.y + threadIdx.y;

    //unrolling kUnrollY*kUnrollX
    auto getRow = [&kPitch, &graph](int row) {
        return (T*)((char*)graph + row * kPitch);
    };
    #pragma unroll kUnrollY
    for (int y = 0; y < kUnrollY; ++y) {
        #pragma unroll kUnrollX
        for (int x = 0; x < kUnrollX; ++x) {
            if (i + y*blockDim.y >= N || j + x*blockDim.x >= N)
                continue;
            const T relax = getRow(i + y*blockDim.y)[k] + getRow(k)[j + x*blockDim.x];
            if (getRow(i + y*blockDim.y)[j + x*blockDim.x] > relax)
                getRow(i + y*blockDim.y)[j + x*blockDim.x] = relax;
        }
    }
}

template <typename T, int UnrollX, int UnrollY>
__global__ void FloydWarshallCore(T *graph, const int N, const size_t kPitch) {
    dim3 block(kBlockSize2D, kBlockSize2D);
    dim3 grid(CeilDiv(N, block.x*UnrollX), CeilDiv(N, block.y*UnrollY));
    for (int k = 0; k < N; ++k) {
        FloydWarshallCoreInnerLoop<T, UnrollX, UnrollY><<<grid, block>>>(graph, N, kPitch, k);
        if (cudaPeekAtLastError() != cudaSuccess) //error has happened, stop here and early return
            return;
    }
}

template <typename T>
void FloydWarshallDriver_(const T *const in_graph, T *out_graph, const int N) {
    //We need to launch exactly `N` child kernel, so check if we need to increase the limit.
    size_t devicePendingKernelLimit;
    CHECK(cudaDeviceGetLimit(&devicePendingKernelLimit, cudaLimitDevRuntimePendingLaunchCount));
    if (devicePendingKernelLimit < N) {
        CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, N));
    }
    size_t pitch = 0;
    CREATE_DEVICE_PTR_2D(T, d_graph, &pitch, N, N);
    CHECK(cudaMemcpy2D(d_graph.get(), pitch, 
        in_graph, sizeof(in_graph[0]) * N/*no pitch*/, sizeof(in_graph[0]) * N, N, cudaMemcpyHostToDevice));
    FloydWarshallCore<T, 2, 2><<<1, 1>>>(d_graph.get(), N, pitch);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy2D(out_graph, sizeof(out_graph[0]) * N/*no pitch*/,
        d_graph.get(), pitch, sizeof(d_graph[0]) * N, N, cudaMemcpyDeviceToHost));
}

/// @brief Find the shortest path of each pairs of node i, j by inline editing the 2D adjacent matrix, `graph`
/// @param graph `graph[i][j]`: the shortest path from node i to j
template <typename T>
std::vector<std::vector<T>> FloydWarshallDriver(std::vector<std::vector<T>> &&graph) {
    if (graph.empty())
        return graph;
    const auto N = graph.size();
    if (std::any_of(graph.begin(), graph.end(), [N](std::vector<T> row) { return row.size() != N; }))
        throw std::invalid_argument("The input adjacent matrix, `graph`, should be a 2D-square");
    std::vector<T> h_graph(N*N);
    //2D to 1D
    for (int i = 0; i < N; ++i) {
        std::copy(graph[i].begin(), graph[i].end(), &h_graph[i*N]);
    }
    FloydWarshallDriver_(h_graph.data(), h_graph.data(), N);
    //1D to 2D
    for (int i = 0; i < N; ++i) {
        std::copy(&h_graph[i*N], &h_graph[i*N] + N, graph[i].data());
    }
    return std::move(graph); //make `graph` rvalue
}
} //namespace graph
} //namespace cualgo_backends
