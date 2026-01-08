// he_kernel.cu
#include <iostream>
#include <hip/hip_runtime.h>


// Define the number of threads per block and blocks per grid
#define BLOCK_SIZE 256
#define GRID_SIZE 128

// Homomorphic encryption parameters (simplified for demonstration)
#define N 1024  // Size of the data array

// Kernel to perform a simple homomorphic encryption operation
// This example uses a trivial operation for demonstration purposes
__global__ void heEncrypt(int* data, int* encrypted_data, int key) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        encrypted_data[idx] = data[idx] + key;  // Simple encryption: add key
    }
}

// Kernel to perform a simple homomorphic decryption operation
// This example uses a trivial operation for demonstration purposes
__global__ void heDecrypt(int* encrypted_data, int* decrypted_data, int key) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        decrypted_data[idx] = encrypted_data[idx] - key;  // Simple decryption: subtract key
    }
}

// Main function to demonstrate the usage of the homomorphic encryption kernels
int main() {
    // Allocate memory on the host
    int* h_data = (int*)malloc(N * sizeof(int));
    int* h_encrypted_data = (int*)malloc(N * sizeof(int));
    int* h_decrypted_data = (int*)malloc(N * sizeof(int));

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }

    // Allocate memory on the device
    int* d_data, *d_encrypted_data, *d_decrypted_data;
    hipError_t err = hipMalloc((void**)&d_data, N * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for d_data (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_encrypted_data, N * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for d_encrypted_data (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_decrypted_data, N * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for d_decrypted_data (error code " << err << ")" << std::endl;
        return -1;
    }

    // Copy data from host to device
    err = hipMemcpy(d_data, h_data, N * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from host to device (error code " << err << ")" << std::endl;
        return -1;
    }

    // Define encryption key
    int key = 10;

    // Launch encryption kernel
    heEncrypt<<<GRID_SIZE, BLOCK_SIZE>>>(d_data, d_encrypted_data, key);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch heEncrypt kernel (error code " << err << ")" << std::endl;
        return -1;
    }

    // Launch decryption kernel
    heDecrypt<<<GRID_SIZE, BLOCK_SIZE>>>(d_encrypted_data, d_decrypted_data, key);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch heDecrypt kernel (error code " << err << ")" << std::endl;
        return -1;
    }

    // Copy results from device to host
    err = hipMemcpy(h_decrypted_data, d_decrypted_data, N * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy data from device to host (error code " << err << ")" << std::endl;
        return -1;
    }

    // Verify the results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_decrypted_data[i] != h_data[i]) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Homomorphic encryption and decryption successful!" << std::endl;
    } else {
        std::cout << "Homomorphic encryption and decryption failed!" << std::endl;
    }

    // Free device memory
    hipFree(d_data);
    hipFree(d_encrypted_data);
    hipFree(d_decrypted_data);

    // Free host memory
    free(h_data);
    free(h_encrypted_data);
    free(h_decrypted_data);

    return 0;
}