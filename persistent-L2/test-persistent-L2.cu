#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include "persistent-L2.h"




// test section
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const* const file, int const line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


__global__ void test_kernel(int* d_target_ptr, size_t size) // we have to write something after read
{
    size_t const idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t const stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < size; i += stride)
    {
        d_target_ptr[i] = d_target_ptr[i] + 1;
    }
}



void launch_test_kernel(int* d_target_ptr, size_t size, cudaStream_t stream)
{
    dim3 const threads_per_block{1024};
    dim3 const blocks_per_grid{1024};
    test_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_target_ptr, size);
    CHECK_LAST_CUDA_ERROR();
}

void one_iteration(int* h_target_ptr, int* d_target_ptr, size_t size, cudaStream_t stream)
{
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_target_ptr, h_target_ptr, size * sizeof(int), cudaMemcpyHostToDevice, stream));
    launch_test_kernel(d_target_ptr, size, stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_target_ptr, d_target_ptr, size * sizeof(int), cudaMemcpyDeviceToHost, stream));
    h_target_ptr[0] = - 1;// write something
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 100,
                          int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

int main(){
    constexpr int const num_repeats{10000};
    constexpr int const num_warmups{100};
    

    cudaDeviceProp device_prop{};
    int current_device{0};
    CHECK_CUDA_ERROR(cudaGetDevice(&current_device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_prop, current_device));
    std::cout << "GPU: " << device_prop.name << std::endl;
    std::cout << "L2 Cache Size: " << device_prop.l2CacheSize / 1024 / 1024
              << " MB" << std::endl;
    std::cout << "Max Persistent L2 Cache Size: "
              << device_prop.persistingL2CacheMaxSize / 1024 / 1024 << " MB"
              << std::endl;


    const size_t size = 1 * 1024 * 1024; // 4 MB

    std::vector<int> target(size, 0);
    
    int* h_target_ptr = target.data(); 
    int* d_target_ptr;
    cudaStream_t normal_stream;
    cudaStream_t persist_stream;

    CHECK_CUDA_ERROR(cudaMalloc(&d_target_ptr, size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&normal_stream));
    CHECK_CUDA_ERROR(cudaStreamCreate(&persist_stream));

    ping_address_to_L2<int>(d_target_ptr, size, &persist_stream);

    // 性能测试
    
    std::function<void(cudaStream_t)> const function{
        std::bind(one_iteration, h_target_ptr, d_target_ptr, size, std::placeholders::_1)};
    std::cout << "[test] for normal" << std::endl;
    float const latency_normal{measure_performance(
        function, normal_stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(4)
              << "Latency Without Using Persistent L2 Cache: " << latency_normal
              << " ms" << std::endl;

    std::cout << "[test] for persist" << std::endl;
    float const latency_persist{measure_performance(
        function, persist_stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(4)
              << "Latency Without Using Persistent L2 Cache: " << latency_persist
              << " ms" << std::endl;


}