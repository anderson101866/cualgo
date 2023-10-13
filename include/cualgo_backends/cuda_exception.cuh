#pragma once
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace cualgo_backends {
class CudaException : public std::logic_error {
    public:
        CudaException(cudaError_t err, const char* file, int line, const char* func)
          : std::logic_error(ComposeErrMsg(err, file, line)),
            err_(err),
            line_(line),
            file_(file),
            func_(func) { }
        
        cudaError_t ErrCode() const { return err_; }
        const char* File() const { return file_; }
        int LineNo()       const { return line_; }
        const char* Func() const { return func_; }
private:
    static std::string ComposeErrMsg(cudaError_t err, const char* file, int line) {
        return std::string("cudaError_t(") + std::to_string(err) + ") - " + cudaGetErrorString(err) + 
               "\n  at " + file + ':' + std::to_string(line);
    }
    cudaError_t err_;
    int line_;
    const char* file_, *func_;
};
} //namespace cualgo_backends 

#ifdef _MSC_VER
    #define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

#ifdef __CUDA_ARCH__
//device code
#define CHECK(call)                                                         \
do {                                                                        \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code: %d, reason: %s\n", error,                             \
                cudaGetErrorString(error));                                 \
    }                                                                       \
} while(0)
#else
//hose code
#define CHECK(call)                                                                  \
do {                                                                                 \
    const cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                      \
        throw cualgo_backends::CudaException(error, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
    }                                                                                \
} while(0)
#endif
