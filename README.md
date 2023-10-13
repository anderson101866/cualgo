# CuAlgo

CuAlgo is a Python library benefiting from GPU-accelerated computing, featuring a collection of fundamental algorithms implemented with CUDA. Currently, it includes the Floyd-Warshall algorithm for graph analysis, showcasing the potential of GPU acceleration.

## Key Features
#### Graph Algorithms: 
 - Floyd-Warshall algorithm

## Why CuAlgo?

- **Significant Speedup**: Experience substantial performance gains with CuAlgo's GPU-accelerated algorithms compared to their CPU approaches.
- **User-Friendly Python Interface**: CuAlgo provides convenient interface for Python users. It is compatible with **NumPy**, allowing for easy data interchange with existing scientific computing workflows. Ensuring that python developers can leverage GPU acceleration without delving into CUDA programming details.
- **Cross-Platform Compatibility**: Developed with CMake, CuAlgo supports cross-platform development, enabling seamless compilation on various operating systems.

## Performance Evaluation
Explore different implementations of the Floyd-Warshall algorithm using datasets of sizes N=40, N=1000, and N=2000. This section presents a comprehensive analysis of the efficiency improvements achieved through GPU acceleration.

### Methodology
- **CPU Version**: The algorithm is executed on the CPU without GPU acceleration.
- **CPU (12 threads) Version**: Runs on the CPU with 12 threads using OpenMP.
- **GPU (Unoptimized) Version**: Initial GPU implementation with similar parallelism as the next GPU (Optimized) Version.
- **GPU (Optimized) Version**: GPU implementation with optimizations, including loop/block unrolling, dynamic parallelism, and coalesced memory access, fully leveraging GPU resources efficiently.

<img src="https://github.com/anderson101866/cualgo/assets/15830675/9d6d4b2e-d4fa-4db1-9a52-fd3d42d325cc" width="600">
<img src="https://github.com/anderson101866/cualgo/assets/15830675/4e3a0fd1-ff81-4d92-9531-b06c1483a9d0" width="600">

The charts illustrate the speedup achieved by CuAlgo's GPU-accelerated algorithms over CPU-based implementations. Notably, the optimized GPU version outperforms both the unoptimized GPU and CPU versions when N grows large, emphasizing the impact of optimization on algorithm efficiency.

### Hardware and Software Information:
| <!--  --> | <!--                            --> |
|-----------|-------------------------------------|
| CPU       | AMD Ryzen 9 5900X 12-Core Processor |
| GPU       | NVIDIA GeForce RTX 3060 Ti - 8GB    |
| RAM       | 32GB DDR4 3600 Mhz                  |
| CUDA Toolkit Version | 12.2                     |
| GPU Driver Version   | 537.13                   |



## Prerequisites
1. GCC compiler with C++ support (gcc works better with CUDA's compiler)
2. GNU Makefile
3. Python 3.7+ with pip available
4. Latest CUDA toolkit installed and nvcc compiler.

**NOTE**: [Recommended] You can skip 3. and 4. by using [conda](https://repo.anaconda.com/archive/).

## Installation
### Linux:
Assume prerequisites 1. and 2. are satisfied.
```bash
conda install cuda -c nvidia
pip install cualgo
```

<!-- ### Windows:
1. [Download Visual Studio with C++ development](https://visualstudio.microsoft.com/en-us/downloads/)2
2. [Install CUDA toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64) (Make sure to install **Runtime**, **Development**, and **Driver**)
3. `pip install cualgo` -->



## Sample Code

Support data type of `Numpy`.
```python
import cualgo
import numpy as np
graph = np.array([
    [0     , 7     , np.inf, 8],
    [np.inf, 0     , 5     , np.inf],
    [np.inf, np.inf, 0     , 2],
    [np.inf, np.inf, np.inf, 0]
], dtype=np.float64)
print(cualgo.floydwarshall(graph))
# [[0.0, 7.0, 12.0, 8.0], [inf, 0.0, 5.0, 7.0], [inf, inf, 0.0, 2.0], [inf, inf, inf, 0.0]]
```

Or just simply pass 2D `list` in python
```python
import cualgo
INF = 9999
graph = [
    [0  , 7  , INF, 8],
    [INF, 0  , 5  , INF],
    [INF, INF, 0  , 2],
    [INF, INF, INF, 0]
]
print(cualgo.floydwarshall(graph))
# [[0, 7, 12, 8], [9999, 0, 5, 7], [9999, 9999, 0, 2], [9999, 9999, 9999, 0]]
```

