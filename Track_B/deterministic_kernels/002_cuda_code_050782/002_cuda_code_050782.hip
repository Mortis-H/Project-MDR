#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#define N 4096
#define BLOCK_SIZE 256

// Kernel to initialize random values for the array
__global__ void initRandom(hiprandState *state, float *data, unsigned int seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        hiprand_init(seed, id, 0, &state[id]);
        data[id] = hiprand_uniform(&state[id]) * 1000.0f;
    }
}

// Kernel to perform a simple sort using Thrust on device memory
__global__ void sortKernel(float *data) {
    // Thrust does not support direct kernel launch, so we use host API calls
    // This is a placeholder to demonstrate the concept. Actual sorting is done in host code.
}

// Error checking macro
#define CUDA_CHECK(call)                                                            \
    {                                                                               \
        hipError_t err = call;                                                     \
        if (err != hipSuccess) {                                                   \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "    \
                      << hipGetErrorString(err) << std::endl;                      \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }

int main() {
    float *d_data;
    hiprandState *d_state;

    // Allocate memory for the random states
    CUDA_CHECK(hipMalloc(&d_state, N * sizeof(hiprandState)));

    // Allocate memory for the data array
    CUDA_CHECK(hipMalloc(&d_data, N * sizeof(float)));

    // Initialize the random number generator
    initRandom<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_state, d_data, 1234);
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipDeviceSynchronize());

    // Copy data to host for sorting with Thrust
    thrust::device_ptr<float> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + N);

    // Copy sorted data back to host for verification (optional)
    float h_data[N];
    CUDA_CHECK(hipMemcpy(h_data, d_data, N * sizeof(float), hipMemcpyDeviceToHost));

    // Free allocated memory
    CUDA_CHECK(hipFree(d_data));
    CUDA_CHECK(hipFree(d_state));

    // Print the first 10 elements of the sorted array for verification
    std::cout << "First 10 elements of the sorted array:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}