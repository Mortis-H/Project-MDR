#include <iostream>
#include <hip/hip_runtime.h>


#define IMAGE_WIDTH 64
#define IMAGE_HEIGHT 64
#define BLOCK_SIZE 16

__global__ void preprocessKernel(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Simple normalization for demonstration
        output[y * width + x] = input[y * width + x] / 255.0f;
    }
}

__global__ void featureExtractionKernel(float* input, float* features, int width, int height) {
    extern __shared__ float sharedData[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (x < width && y < height) {
        // Load data into shared memory
        sharedData[ty * blockDim.x + tx] = input[y * width + x];
        __syncthreads();

        // Example feature extraction: compute average in block
        float sum = 0.0f;
        for (int i = 0; i < blockDim.y; ++i) {
            for (int j = 0; j < blockDim.x; ++j) {
                sum += sharedData[i * blockDim.x + j];
            }
        }
        __syncthreads();

        // Write feature to global memory
        if (tx == 0 && ty == 0) {
            features[blockIdx.y * gridDim.x + blockIdx.x] = sum / (blockDim.x * blockDim.y);
        }
    }
}

void checkCudaError(hipError_t error, const char* file, int line) {
    if (error != hipSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " - " << hipGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int width = IMAGE_WIDTH;
    const int height = IMAGE_HEIGHT;
    const int imageSize = width * height;
    const int numBlocksX = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int numBlocksY = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* h_input = new float[imageSize];
    float* h_features = new float[numBlocksX * numBlocksY];

    // Initialize input image data
    for (int i = 0; i < imageSize; ++i) {
        h_input[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 255.0f;
    }

    float* d_input;
    float* d_features;
    size_t inputSize = imageSize * sizeof(float);
    size_t featuresSize = (numBlocksX * numBlocksY) * sizeof(float);

    checkCudaError(hipMalloc((void**)&d_input, inputSize), __FILE__, __LINE__);
    checkCudaError(hipMalloc((void**)&d_features, featuresSize), __FILE__, __LINE__);

    checkCudaError(hipMemcpy(d_input, h_input, inputSize, hipMemcpyHostToDevice), __FILE__, __LINE__);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(numBlocksX, numBlocksY);

    // Launch preprocess kernel
    preprocessKernel<<<gridSize, blockSize>>>(d_input, d_input, width, height);
    checkCudaError(hipGetLastError(), __FILE__, __LINE__);
    checkCudaError(hipDeviceSynchronize(), __FILE__, __LINE__);

    // Launch feature extraction kernel with dynamic parallelism
    featureExtractionKernel<<<gridSize, blockSize, blockSize.x * blockSize.y * sizeof(float)>>>(d_input, d_features, width, height);
    checkCudaError(hipGetLastError(), __FILE__, __LINE__);
    checkCudaError(hipDeviceSynchronize(), __FILE__, __LINE__);

    // Copy results back to host
    checkCudaError(hipMemcpy(h_features, d_features, featuresSize, hipMemcpyDeviceToHost), __FILE__, __LINE__);

    // Output some features for demonstration
    for (int i = 0; i < 10 && i < numBlocksX * numBlocksY; ++i) {
        std::cout << "Feature " << i << ": " << h_features[i] << std::endl;
    }

    // Clean up
    delete[] h_input;
    delete[] h_features;
    checkCudaError(hipFree(d_input), __FILE__, __LINE__);
    checkCudaError(hipFree(d_features), __FILE__, __LINE__);

    return 0;
}