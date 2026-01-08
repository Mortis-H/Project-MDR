#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define N 1024  // Number of neurons per layer

// Kernel to perform matrix multiplication (simulating forward pass)
__global__ void matrixMultiply(float *A, float *B, float *C, int width) {
    __shared__ float sA[16][17];  // Padding to avoid bank conflicts
    __shared__ float sB[16][16];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * 16 + ty;
    int Col = bx * 16 + tx;

    float Cvalue = 0;

    // Loop over the A and B tiles required to compute the C element
    for (int m = 0; m < (width - 1) / 16 + 1; ++m) {
        // Load the matrices from device memory to shared memory
        if (Row < width && m * 16 + tx < width)
            sA[ty][tx] = A[Row * width + m * 16 + tx];
        else
            sA[ty][tx] = 0.0;

        if (Col < width && m * 16 + ty < width)
            sB[ty][tx] = B[(m * 16 + ty) * width + Col];
        else
            sB[ty][tx] = 0.0;

        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element of the block sub-matrix
        for (int e = 0; e < 16; ++e)
            Cvalue += sA[ty][e] * sB[e][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    if (Row < width && Col < width)
        C[Row * width + Col] = Cvalue;
}

// Kernel to perform element-wise addition (simulating bias addition)
__global__ void addBias(float *A, float *bias, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        A[idx] += bias[idx];
    }
}

// Kernel to perform element-wise activation function (ReLU)
__global__ void relu(float *A, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        A[idx] = fmaxf(0.0f, A[idx]);
    }
}

int main() {
    float *h_A, *h_B, *h_C, *h_bias;
    float *d_A, *d_B, *d_C, *d_bias;
    int width = N;

    // Allocate host memory
    h_A = (float *)malloc(width * width * sizeof(float));
    h_B = (float *)malloc(width * width * sizeof(float));
    h_C = (float *)malloc(width * width * sizeof(float));
    h_bias = (float *)malloc(width * sizeof(float));

    // Initialize host matrices and bias
    for (int i = 0; i < width * width; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < width; i++) {
        h_bias[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    hipError_t err = hipMalloc((void **)&d_A, width * width * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for d_A (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_B, width * width * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for d_B (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_C, width * width * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for d_C (error code " << err << ")" << std::endl;
        return -1;
    }
    err = hipMalloc((void **)&d_bias, width * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device memory for d_bias (error code " << err << ")" << std::endl;
        return -1;
    }

    // Copy matrices from host memory to device memory
    hipMemcpy(d_A, h_A, width * width * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, width * width * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_bias, h_bias, width * sizeof(float), hipMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Perform matrix multiplication (simulating forward pass)
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch matrixMultiply kernel (error code " << err << ")" << std::endl;
        return -1;
    }

    // Perform bias addition
    int threadsPerBlockAdd = 256;
    int blocksPerGridAdd = (width + threadsPerBlockAdd - 1) / threadsPerBlockAdd;
    addBias<<<blocksPerGridAdd, threadsPerBlockAdd>>>(d_C, d_bias, width);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch addBias kernel (error code " << err << ")" << std::endl;
        return -1;
    }

    // Perform ReLU activation
    relu<<<blocksPerGridAdd, threadsPerBlockAdd>>>(d_C, width);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch relu kernel (error code " << err << ")" << std::endl;
        return -1;
    }

    // Copy result from device memory to host memory
    hipMemcpy(h_C, d_C, width * width * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_bias);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_bias);

    return 0;
}