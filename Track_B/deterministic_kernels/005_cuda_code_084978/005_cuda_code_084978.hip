#include <iostream>
#include <hip/hip_runtime.h>

#include <hiprand/hiprand_kernel.h>
#include <hip/hip_fp16.h>

#define BLOCK_SIZE 32

__global__ void initializeMatrix(float *d_matrix, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        hiprandState state;
        hiprand_init((unsigned long long)clock() + row, col, 0, &state);
        d_matrix[row * width + col] = hiprand_uniform(&state) * 10.0f;
    }
}

__global__ void meanSubtraction(float *d_matrix, float *d_mean, int width, int height) {
    __shared__ float cache[BLOCK_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (row < height) {
        for (int i = 0; i < width; ++i) {
            sum += d_matrix[row * width + i];
        }
        cache[threadIdx.x] = sum / width;
        __syncthreads();

        if (threadIdx.x == 0) {
            atomicAdd(d_mean + blockIdx.y, cache[0]);
        }
    }
}

__global__ void subtractMean(float *d_matrix, float *d_mean, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        d_matrix[row * width + col] -= d_mean[row];
    }
}

__global__ void covarianceMatrix(float *d_matrix, float *d_cov, int width, int height) {
    __shared__ float sharedMatrix[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (row < width && col < height) {
        for (int i = 0; i < width; i += BLOCK_SIZE) {
            if (i + threadIdx.x < width) {
                sharedMatrix[threadIdx.y][threadIdx.x] = d_matrix[(i + threadIdx.x) * width + row];
            } else {
                sharedMatrix[threadIdx.y][threadIdx.x] = 0.0f;
            }
            __syncthreads();

            for (int j = 0; j < BLOCK_SIZE; ++j) {
                sum += sharedMatrix[threadIdx.y][j] * d_matrix[col * width + (i + j)];
            }
            __syncthreads();
        }
        d_cov[row * width + col] = sum / (width - 1);
    }
}

__global__ void eigenvectorKernel(float *d_cov, float *d_eigenvectors, int width) {
    // Placeholder for eigenvalue decomposition kernel
    // This is a simplified version for demonstration
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        d_eigenvectors[row * width + col] = (row == col) ? 1.0f : 0.0f;
    }
}

void checkCudaError(hipError_t err, const char *operation) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << operation << " failed: " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int width = 32;
    int height = 32;
    size_t size = width * height * sizeof(float);

    float *d_matrix, *d_mean, *d_cov, *d_eigenvectors;
    checkCudaError(hipMalloc(&d_matrix, size), "hipMalloc d_matrix");
    checkCudaError(hipMalloc(&d_mean, height * sizeof(float)), "hipMalloc d_mean");
    checkCudaError(hipMalloc(&d_cov, size), "hipMalloc d_cov");
    checkCudaError(hipMalloc(&d_eigenvectors, size), "hipMalloc d_eigenvectors");

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    initializeMatrix<<<gridSize, blockSize>>>(d_matrix, width, height);
    checkCudaError(hipGetLastError(), "initializeMatrix kernel launch");

    hipMemset(d_mean, 0, height * sizeof(float));
    meanSubtraction<<<gridSize, blockSize>>>(d_matrix, d_mean, width, height);
    checkCudaError(hipGetLastError(), "meanSubtraction kernel launch");
    hipDeviceSynchronize();

    subtractMean<<<gridSize, blockSize>>>(d_matrix, d_mean, width, height);
    checkCudaError(hipGetLastError(), "subtractMean kernel launch");

    hipMemset(d_cov, 0, size);
    covarianceMatrix<<<gridSize, blockSize>>>(d_matrix, d_cov, width, height);
    checkCudaError(hipGetLastError(), "covarianceMatrix kernel launch");

    eigenvectorKernel<<<gridSize, blockSize>>>(d_cov, d_eigenvectors, width);
    checkCudaError(hipGetLastError(), "eigenvectorKernel kernel launch");

    checkCudaError(hipFree(d_matrix), "hipFree d_matrix");
    checkCudaError(hipFree(d_mean), "hipFree d_mean");
    checkCudaError(hipFree(d_cov), "hipFree d_cov");
    checkCudaError(hipFree(d_eigenvectors), "hipFree d_eigenvectors");

    return 0;
}