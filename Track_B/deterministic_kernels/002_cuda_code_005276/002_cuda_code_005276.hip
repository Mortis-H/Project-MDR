#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <algorithm>

#define BLOCK_SIZE 256

__global__ void parallelReduce(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    // Load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Reduce in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

__global__ void medianKernel(float *d_sorted, float *d_median, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load shared mem
    if (idx < n) {
        sdata[tid] = d_sorted[idx];
    }
    __syncthreads();

    // Find median
    if (tid == 0) {
        int mid = n / 2;
        if (n % 2 == 0) {
            d_median[0] = (sdata[mid - 1] + sdata[mid]) / 2.0f;
        } else {
            d_median[0] = sdata[mid];
        }
    }
}

int main() {
    const int n = 16384;
    thrust::host_vector<float> h_input(n);
    std::generate(h_input.begin(), h_input.end(), rand);
    thrust::device_vector<float> d_input = h_input;

    // Allocate memory for intermediate results
    thrust::device_vector<float> d_intermediate((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Perform parallel reduction to sum all elements
    parallelReduce<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_intermediate.data()), n);
    hipDeviceSynchronize();

    // Check for errors in kernel launch
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Allocate memory for median
    float *d_median;
    hipMalloc(&d_median, sizeof(float));

    // Sort the array to find the median
    thrust::sort(d_input.begin(), d_input.end());

    // Find the median
    medianKernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(thrust::raw_pointer_cast(d_input.data()), d_median, n);
    hipDeviceSynchronize();

    // Check for errors in kernel launch
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result back to host
    float h_median;
    hipMemcpy(&h_median, d_median, sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_median);

    // Output the result
    std::cout << "Median: " << h_median << std::endl;

    return 0;
}