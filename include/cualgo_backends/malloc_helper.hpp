//A header pre-define shortcuts to allocate device memory in RAII syntax

#pragma once
#include <memory>
#include "cualgo_backends/cuda_exception.hpp"

#define _CONCAT(a, b) _CONCAT_IMPL(a, b)
#define _CONCAT_IMPL(a, b) a ## b
#define _UNIQUE_NAME(base) _CONCAT(base, __LINE__)

/**
 * @brief Similar to the following but create memory on device instead
 * @note The memory is freed when stack is unwound. 
 * As pybind always catch c++ exception to throw suitable python exception. 
 * It's safe we expects pybind11 always handle excetion and hence stack will be unwound.
 * @code{.cpp}
 * unique_ptr<T[]> ptr = ...;
 * if (!ptr)
 *     throw CudaException(...)
 * @endcode
 * @tparam T template <typename T> for the type of returned pointer
 * @param[out] device_ptr name of the declared unique_ptr
 * @param[out] p_pitch    pointer to the returned pitch
 * @param[in]  width      actual number of element in each row (NOT in bytes)
 * @param[in]  height     how many column
 * @note this should be macro instead of template function to keep the __FILE__, __LINE__ for debugging
*/
#define CREATE_DEVICE_PTR_2D(T, device_ptr, p_pitch, width, height) \
    T *_UNIQUE_NAME(ptr) = nullptr; \
    { \
        const cudaError_t error = cudaMallocPitch(&_UNIQUE_NAME(ptr), p_pitch, sizeof(T) * width, (height)); \
        if (error != cudaSuccess) \
            throw cualgo_backends::CudaException(error, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
    } \
    auto _UNIQUE_NAME(deleter) = [](T *ptr) { \
        const cudaError_t error = cudaFree(ptr); \
        if (error != cudaSuccess) \
            throw cualgo_backends::CudaException(error, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
    }; \
    std::unique_ptr<T[], decltype(_UNIQUE_NAME(deleter))> device_ptr(_UNIQUE_NAME(ptr), _UNIQUE_NAME(deleter));
