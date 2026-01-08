#include <iostream>
#include <hip/hip_runtime.h>

#include <hiprand/hiprand_kernel.h>

#define BLOCK_SIZE 16
#define MAX_ITERATIONS 100
#define EPSILON 1e-5

// Kernel to initialize random noise for demonstration purposes
__global__ void initRandomNoise(float* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * width + idx;

    if (idx < width && idy < height) {
        hiprandState state;
        hiprand_init((unsigned long long)clock() + index, 0, 0, &state);
        image[index] = hiprand_uniform(&state);
    }
}

// Active contour kernel
__global__ void activeContourKernel(float* image, float* contour, float* newContour, int width, int height, int numPoints) {
    extern __shared__ float sharedImage[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * width + idx;

    // Load image data into shared memory
    if (idx < width && idy < height) {
        sharedImage[threadIdx.y * blockDim.x + threadIdx.x] = image[index];
    }
    __syncthreads();

    // Only process contour points
    if (idx < numPoints) {
        int x = (int)contour[idx * 2];
        int y = (int)contour[idx * 2 + 1];

        // Boundary check
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
            // Fetch neighbors from shared memory
            float c = sharedImage[y * width + x];
            float n = sharedImage[(y - 1) * width + x];
            float s = sharedImage[(y + 1) * width + x];
            float e = sharedImage[y * width + x + 1];
            float w = sharedImage[y * width + x - 1];

            // Simple gradient descent to move contour
            float dx = e - w;
            float dy = s - n;
            float magnitude = sqrtf(dx * dx + dy * dy) + EPSILON;

            newContour[idx * 2] = x + dx / magnitude;
            newContour[idx * 2 + 1] = y + dy / magnitude;
        }
    }
}

int main() {
    int width = 256;
    int height = 256;
    int numPoints = 100;

    // Allocate host memory
    float* h_image = new float[width * height];
    float* h_contour = new float[numPoints * 2];
    float* h_newContour = new float[numPoints * 2];

    // Initialize contour points randomly within the image
    for (int i = 0; i < numPoints; ++i) {
        h_contour[i * 2] = rand() % width;
        h_contour[i * 2 + 1] = rand() % height;
    }

    // Allocate device memory
    float* d_image;
    float* d_contour;
    float* d_newContour;
    hipMalloc((void**)&d_image, width * height * sizeof(float));
    hipMalloc((void**)&d_contour, numPoints * 2 * sizeof(float));
    hipMalloc((void**)&d_newContour, numPoints * 2 * sizeof(float));

    // Copy data to device
    hipMemcpy(d_contour, h_contour, numPoints * 2 * sizeof(float), hipMemcpyHostToDevice);

    // Initialize random noise in the image
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    initRandomNoise<<<gridSize, blockSize>>>(d_image, width, height);
    hipDeviceSynchronize();

    // Perform active contour iterations
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        activeContourKernel<<<gridSize, blockSize, blockSize.x * blockSize.y * sizeof(float)>>>(
            d_image, d_contour, d_newContour, width, height, numPoints);
        hipDeviceSynchronize();

        // Swap contours
        float* temp = d_contour;
        d_contour = d_newContour;
        d_newContour = temp;
    }

    // Copy final contour back to host
    hipMemcpy(h_contour, d_contour, numPoints * 2 * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_image);
    hipFree(d_contour);
    hipFree(d_newContour);

    // Free host memory
    delete[] h_image;
    delete[] h_contour;
    delete[] h_newContour;

    return 0;
}