#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define BLOCK_SIZE 16
#define GRID_SIZE 16

// Kernel to compute divergence of fluid velocity field
__global__ void computeDivergence(float* velocity, float* divergence, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1 && z > 0 && z < depth - 1) {
        int idx = x + y * width + z * width * height;
        divergence[idx] = 
            (velocity[idx + 1] - velocity[idx - 1]) +
            (velocity[idx + width] - velocity[idx - width]) +
            (velocity[idx + width * height] - velocity[idx - width * height]);
    }
}

// Kernel to reduce divergence
__global__ void reduceDivergence(float* divergence, float* output, int size) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load shared mem
    sdata[tid] = (i < size) ? divergence[i] : 0.0f;
    __syncthreads();

    // Reduce
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
    const int width = 32;
    const int height = 32;
    const int depth = 32;
    const int numElements = width * height * depth;
    const int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* h_velocity = new float[numElements];
    float* h_divergence = new float[numElements];
    float* h_output = new float[numBlocks];

    // Initialize velocity field
    for (int i = 0; i < numElements; ++i) {
        h_velocity[i] = 0.1f * i; // Example initialization
    }

    float* d_velocity;
    float* d_divergence;
    float* d_output;

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_velocity, numElements * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_velocity: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_divergence, numElements * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_divergence: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_output, numBlocks * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating d_output: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Asynchronous memory copy from host to device
    hipMemcpyAsync(d_velocity, h_velocity, numElements * sizeof(float), hipMemcpyHostToDevice, 0);

    // Define grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, (depth + blockSize.z - 1) / blockSize.z);

    // Launch computeDivergence kernel
    computeDivergence<<<gridSize, blockSize>>>(d_velocity, d_divergence, width, height, depth);

    // Asynchronous memory copy from device to host
    hipMemcpyAsync(h_divergence, d_divergence, numElements * sizeof(float), hipMemcpyDeviceToHost, 0);

    // Launch reduceDivergence kernel
    reduceDivergence<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_divergence, d_output, numElements);

    // Synchronize to ensure all operations are complete
    hipDeviceSynchronize();

    // Copy final result to host
    hipMemcpy(h_output, d_output, numBlocks * sizeof(float), hipMemcpyDeviceToHost);

    // Sum up the final results
    float totalDivergence = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        totalDivergence += h_output[i];
    }

    std::cout << "Total Divergence: " << totalDivergence << std::endl;

    // Free device memory
    hipFree(d_velocity);
    hipFree(d_divergence);
    hipFree(d_output);

    // Free host memory
    delete[] h_velocity;
    delete[] h_divergence;
    delete[] h_output;

    return 0;
}