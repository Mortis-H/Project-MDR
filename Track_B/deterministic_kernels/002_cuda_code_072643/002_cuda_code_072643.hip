#include <iostream>
#include <hip/hip_runtime.h>
#include <cmath>

// Define the size of the array
#define N 16384

// CUDA kernel to compute the dot product of two vectors using warp-level reduction
__global__ void dotProductKernel(const float* a, const float* b, float* result) {
    extern __shared__ float sharedData[];

    // Calculate the thread index
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory
    sharedData[tid] = 0.0f;
    __syncthreads();

    // Accumulate dot product in shared memory
    if (i < N) {
        sharedData[tid] = a[i] * b[i];
    }
    __syncthreads();

    // Warp-level reduction to sum the values in shared memory
    for (int stride = warpSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result to global memory if this is the first thread in the block
    if (tid == 0) {
        atomicAdd(result, sharedData[0]);
    }
}

// CUDA kernel to compute the magnitude of a vector
__global__ void magnitudeKernel(const float* v, float* result) {
    extern __shared__ float sharedData[];

    // Calculate the thread index
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory
    sharedData[tid] = 0.0f;
    __syncthreads();

    // Accumulate the square of the vector elements in shared memory
    if (i < N) {
        sharedData[tid] = v[i] * v[i];
    }
    __syncthreads();

    // Warp-level reduction to sum the values in shared memory
    for (int stride = warpSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result to global memory if this is the first thread in the block
    if (tid == 0) {
        atomicAdd(result, sharedData[0]);
    }
}

int main() {
    // Allocate host memory
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_dotProduct = new float[1];
    float* h_magnitudeA = new float[1];
    float* h_magnitudeB = new float[1];
    *h_dotProduct = 0.0f;
    *h_magnitudeA = 0.0f;
    *h_magnitudeB = 0.0f;

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float* d_a;
    float* d_b;
    float* d_dotProduct;
    float* d_magnitudeA;
    float* d_magnitudeB;
    hipMalloc((void**)&d_a, N * sizeof(float));
    hipMalloc((void**)&d_b, N * sizeof(float));
    hipMalloc((void**)&d_dotProduct, sizeof(float));
    hipMalloc((void**)&d_magnitudeA, sizeof(float));
    hipMalloc((void**)&d_magnitudeB, sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_dotProduct, h_dotProduct, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_magnitudeA, h_magnitudeA, sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_magnitudeB, h_magnitudeB, sizeof(float), hipMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernels
    dotProductKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_a, d_b, d_dotProduct);
    magnitudeKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_a, d_magnitudeA);
    magnitudeKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_b, d_magnitudeB);

    // Check for errors in kernel launch
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    hipMemcpy(h_dotProduct, d_dotProduct, sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_magnitudeA, d_magnitudeA, sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_magnitudeB, d_magnitudeB, sizeof(float), hipMemcpyDeviceToHost);

    // Compute cosine similarity
    float cosineSimilarity = *h_dotProduct / (sqrt(*h_magnitudeA) * sqrt(*h_magnitudeB));

    // Print the result
    std::cout << "Cosine Similarity: " << cosineSimilarity << std::endl;

    // Free device memory
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_dotProduct);
    hipFree(d_magnitudeA);
    hipFree(d_magnitudeB);

    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_dotProduct;
    delete[] h_magnitudeA;
    delete[] h_magnitudeB;

    return 0;
}