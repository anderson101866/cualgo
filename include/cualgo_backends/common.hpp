#pragma once
#include <cuda_runtime.h>
#include <cassert>

namespace cualgo_backends {
    /// @brief equivalent to `int x, y; ceil(x/y);`
    __host__ __device__ inline int CeilDiv(const int x, const int y) {
        return (x+y-1) / y;
    }

    constexpr int kBlockSize2D = 16;
    constexpr int kBlockSize1D = 512;
}
