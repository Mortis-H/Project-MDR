#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <hiprand/hiprand_kernel.h>

#define N 8192

// Kernel to initialize random numbers using curand
__global__ void initRandom(hiprandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N) {
        hiprand_init(seed, id, 0, &state[id]);
    }
}

// Kernel to generate random numbers
__global__ void generateRandom(hiprandState *state, float *data) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N) {
        data[id] = hiprand_uniform(&state[id]);
    }
}

// Kernel to perform bitonic sort
__global__ void bitonicSort(float *data, int j, int k) {
    unsigned int i, ixj;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        ixj = i ^ j;
        if (ixj > i) {
            if ((i & k) == 0) {
                if (data[i] > data[ixj]) {
                    float temp = data[i];
                    data[i] = data[ixj];
                    data[ixj] = temp;
                }
            } else {
                if (data[i] < data[ixj]) {
                    float temp = data[i];
                    data[i] = data[ixj];
                    data[ixj] = temp;
                }
            }
        }
    }
}

// Function to check CUDA errors
void checkCudaError(hipError_t err, const char *msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    float *d_data;
    hiprandState *d_state;

    // Allocate device memory for data and random states
    checkCudaError(hipMalloc(&d_data, N * sizeof(float)), "hipMalloc d_data");
    checkCudaError(hipMalloc(&d_state, N * sizeof(hiprandState)), "hipMalloc d_state");

    // Initialize random number generator states
    initRandom<<<(N + 255) / 256, 256>>>(d_state, 1234);
    checkCudaError(hipGetLastError(), "initRandom kernel launch failed");
    checkCudaError(hipDeviceSynchronize(), "initRandom kernel sync failed");

    // Generate random numbers
    generateRandom<<<(N + 255) / 256, 256>>>(d_state, d_data);
    checkCudaError(hipGetLastError(), "generateRandom kernel launch failed");
    checkCudaError(hipDeviceSynchronize(), "generateRandom kernel sync failed");

    // Perform bitonic sort
    for (int j = 2; j <= N; j *= 2) {
        for (int k = j / 2; k > 0; k /= 2) {
            bitonicSort<<<(N + 255) / 256, 256>>>(d_data, j, k);
            checkCudaError(hipGetLastError(), "bitonicSort kernel launch failed");
            checkCudaError(hipDeviceSynchronize(), "bitonicSort kernel sync failed");
        }
    }

    // Copy sorted data back to host
    thrust::device_ptr<float> dev_ptr(d_data);
    thrust::host_vector<float> h_data(dev_ptr, dev_ptr + N);

    // Print first 10 elements of sorted array for verification
    std::cout << "First 10 elements of sorted array:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    checkCudaError(hipFree(d_data), "hipFree d_data");
    checkCudaError(hipFree(d_state), "hipFree d_state");

    return 0;
}