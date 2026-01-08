#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include <cmath>

namespace cg = cooperative_groups;

#define BLOCK_SIZE 16
#define FILTER_SIZE 13
#define SIGMA_D 10.0f
#define SIGMA_R 0.1f

__device__ float gaussian(float x, float sigma) {
    return expf(-(x * x) / (2 * sigma * sigma)) / (sigma * sqrtf(2 * M_PI));
}

__device__ float bilateral_filter_kernel(float* img, int width, int height, int x, int y, float* filter) {
    float sum = 0.0f;
    float norm = 0.0f;
    int halfSize = FILTER_SIZE / 2;

    for (int i = -halfSize; i <= halfSize; ++i) {
        for (int j = -halfSize; j <= halfSize; ++j) {
            int px = x + i;
            int py = y + j;

            if (px >= 0 && px < width && py >= 0 && py < height) {
                float intensityDifference = img[py * width + px] - img[y * width + x];
                float spatialWeight = filter[(i + halfSize) * FILTER_SIZE + (j + halfSize)];
                float rangeWeight = gaussian(intensityDifference, SIGMA_R);
                float weight = spatialWeight * rangeWeight;
                sum += img[py * width + px] * weight;
                norm += weight;
            }
        }
    }

    return sum / norm;
}

__global__ void precompute_filter(float* filter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int x = idx - FILTER_SIZE / 2;
    int y = idy - FILTER_SIZE / 2;
    filter[idx * FILTER_SIZE + idy] = gaussian(sqrtf(x * x + y * y), SIGMA_D);
}

__global__ void bilateral_filter(float* input, float* output, float* filter, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        output[y * width + x] = bilateral_filter_kernel(input, width, height, x, y, filter);
    }
}

int main() {
    int width = 512;
    int height = 512;
    size_t imgSize = width * height * sizeof(float);
    size_t filterSize = FILTER_SIZE * FILTER_SIZE * sizeof(float);

    float* h_input = new float[width * height];
    float* h_output = new float[width * height];
    float* h_filter = new float[FILTER_SIZE * FILTER_SIZE];

    // Initialize input image with random values
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* d_input;
    float* d_output;
    float* d_filter;

    hipError_t err = hipMalloc(&d_input, imgSize);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device input memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc(&d_output, imgSize);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device output memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMalloc(&d_filter, filterSize);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate device filter memory: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMemcpy(d_input, h_input, imgSize, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy input data to device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((FILTER_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (FILTER_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);

    precompute_filter<<<blocksPerGrid, threadsPerBlock>>>(d_filter);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch precompute_filter kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize after precompute_filter: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    blocksPerGrid = dim3((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    bilateral_filter<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_filter, width, height);
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Failed to launch bilateral_filter kernel: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "Failed to synchronize after bilateral_filter: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    err = hipMemcpy(h_output, d_output, imgSize, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Failed to copy output data from device: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_filter);

    delete[] h_input;
    delete[] h_output;
    delete[] h_filter;

    std::cout << "Bilateral filter completed successfully." << std::endl;

    return 0;
}