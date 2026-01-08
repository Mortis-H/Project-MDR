#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#define BLOCK_SIZE 16

// Kernel to initialize the image with random values
__global__ void initializeImage(unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        hiprandState state;
        hiprand_init(clock64(), x * y, 0, &state);
        image[y * width + x] = static_cast<unsigned char>(hiprand(&state) % 256);
    }
}

// Kernel for region growing
__global__ void regionGrowing(unsigned char* image, int* labels, int width, int height, int seedX, int seedY, unsigned char threshold) {
    extern __shared__ unsigned char sharedImage[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    // Load image data into shared memory
    if (x < width && y < height) {
        sharedImage[threadIdx.y * blockDim.x + threadIdx.x] = image[idx];
    }
    __syncthreads();

    // Check if the current pixel is within the region
    if (x < width && y < height) {
        int sharedIdx = threadIdx.y * blockDim.x + threadIdx.x;
        if (abs(sharedImage[sharedIdx] - sharedImage[0]) <= threshold) {
            labels[idx] = 1; // Mark as part of the region
        }
    }
}

int main() {
    int width = 256;
    int height = 256;
    int size = width * height;
    unsigned char* h_image;
    int* h_labels;
    unsigned char* d_image;
    int* d_labels;

    // Allocate host memory
    h_image = new unsigned char[size];
    h_labels = new int[size];

    // Allocate device memory
    hipError_t err = hipMalloc((void**)&d_image, size * sizeof(unsigned char));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for image: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    err = hipMalloc((void**)&d_labels, size * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for labels: " << hipGetErrorString(err) << std::endl;
        hipFree(d_image);
        return -1;
    }

    // Initialize image with random values
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    hiprandState* devStates;
    hipMalloc(&devStates, size * sizeof(hiprandState));
    initializeImage<<<gridSize, blockSize>>>(d_image, width, height);
    hipFree(devStates);

    // Set seed point and threshold
    int seedX = width / 2;
    int seedY = height / 2;
    unsigned char threshold = 30;

    // Launch region growing kernel
    regionGrowing<<<gridSize, blockSize, blockSize.x * blockSize.y * sizeof(unsigned char)>>>(d_image, d_labels, width, height, seedX, seedY, threshold);

    // Copy results back to host
    err = hipMemcpy(h_labels, d_labels, size * sizeof(int), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying labels from device to host: " << hipGetErrorString(err) << std::endl;
        hipFree(d_image);
        hipFree(d_labels);
        delete[] h_image;
        delete[] h_labels;
        return -1;
    }

    // Free device memory
    hipFree(d_image);
    hipFree(d_labels);

    // Free host memory
    delete[] h_image;
    delete[] h_labels;

    std::cout << "Region growing completed successfully." << std::endl;

    return 0;
}