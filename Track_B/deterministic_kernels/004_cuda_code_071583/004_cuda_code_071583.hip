#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

// Define the size of the array to sort
#define ARRAY_SIZE 8192

// CUDA kernel for sorting using thrust
__global__ void sortKernel(float* d_array) {
    // Thrust does not require explicit kernel launches for sorting
    // Sorting is handled by thrust::sort which can be called from host
    // This kernel is a placeholder for demonstration purposes
}

// Function to check CUDA errors
void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Allocate and initialize host array
    float h_array[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        h_array[i] = static_cast<float>(rand()) / RAND_MAX; // Random values between 0 and 1
    }

    // Allocate device array
    float* d_array;
    checkCudaError(hipMalloc((void**)&d_array, ARRAY_SIZE * sizeof(float)), "hipMalloc failed");

    // Copy data from host to device
    checkCudaError(hipMemcpy(d_array, h_array, ARRAY_SIZE * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy failed");

    // Launch sortKernel (though thrust::sort will be used directly from host)
    sortKernel<<<1, 1>>>(d_array);
    checkCudaError(hipGetLastError(), "Kernel launch failed");
    checkCudaError(hipDeviceSynchronize(), "Device synchronize failed");

    // Perform sorting using thrust
    thrust::device_ptr<float> d_ptr(d_array);
    thrust::sort(d_ptr, d_ptr + ARRAY_SIZE);

    // Copy sorted data back to host
    checkCudaError(hipMemcpy(h_array, d_array, ARRAY_SIZE * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy failed");

    // Verify the result
    for (int i = 0; i < ARRAY_SIZE - 1; ++i) {
        if (h_array[i] > h_array[i + 1]) {
            std::cerr << "Sorting failed: element " << i << " is greater than element " << i + 1 << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Free device memory
    checkCudaError(hipFree(d_array), "hipFree failed");

    std::cout << "Array sorted successfully using thrust on a Turing architecture." << std::endl;

    return 0;
}