#include <iostream>
#include <hip/hip_runtime.h>

// Define some constants for the example
#define BATCH_SIZE 1
#define INPUT_CHANNELS 1
#define INPUT_HEIGHT 28
#define INPUT_WIDTH 28
#define FILTER_SIZE 3
#define NUM_FILTERS 16
#define OUTPUT_CHANNELS NUM_FILTERS
#define OUTPUT_HEIGHT (INPUT_HEIGHT - FILTER_SIZE + 1)
#define OUTPUT_WIDTH (INPUT_WIDTH - FILTER_SIZE + 1)

// Kernel for convolution operation
__global__ void convKernel(float* input, float* filters, float* output, int inputHeight, int inputWidth, int filterSize, int numFilters) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;

    if (x < inputWidth - filterSize + 1 && y < inputHeight - filterSize + 1 && f < numFilters) {
        float sum = 0.0f;
        for (int ky = 0; ky < filterSize; ky++) {
            for (int kx = 0; kx < filterSize; kx++) {
                int inputIdx = (y + ky) * inputWidth + (x + kx);
                int filterIdx = f * filterSize * filterSize + ky * filterSize + kx;
                sum += input[inputIdx] * filters[filterIdx];
            }
        }
        int outputIdx = f * OUTPUT_HEIGHT * OUTPUT_WIDTH + y * OUTPUT_WIDTH + x;
        output[outputIdx] = sum;
    }
}

// Kernel for backpropagation through convolution layer
__global__ void convBackpropKernel(float* dOutput, float* filters, float* dInput, int inputHeight, int inputWidth, int filterSize, int numFilters) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;

    if (x < inputWidth && y < inputHeight && f < numFilters) {
        float sum = 0.0f;
        for (int ky = 0; ky < filterSize; ky++) {
            for (int kx = 0; kx < filterSize; kx++) {
                if (y - ky >= 0 && x - kx >= 0) {
                    int outputIdx = f * OUTPUT_HEIGHT * OUTPUT_WIDTH + (y - ky) * OUTPUT_WIDTH + (x - kx);
                    int filterIdx = f * filterSize * filterSize + ky * filterSize + kx;
                    sum += dOutput[outputIdx] * filters[filterIdx];
                }
            }
        }
        int inputIdx = y * inputWidth + x;
        dInput[inputIdx] = sum;
    }
}

int main() {
    // Allocate and initialize memory on the host
    float* h_input = new float[INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH];
    float* h_filters = new float[NUM_FILTERS * FILTER_SIZE * FILTER_SIZE];
    float* h_output = new float[NUM_FILTERS * OUTPUT_HEIGHT * OUTPUT_WIDTH];
    float* h_dOutput = new float[NUM_FILTERS * OUTPUT_HEIGHT * OUTPUT_WIDTH];
    float* h_dInput = new float[INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH];

    // Initialize with some values
    for (int i = 0; i < INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH; i++) h_input[i] = 1.0f;
    for (int i = 0; i < NUM_FILTERS * FILTER_SIZE * FILTER_SIZE; i++) h_filters[i] = 0.5f;
    for (int i = 0; i < NUM_FILTERS * OUTPUT_HEIGHT * OUTPUT_WIDTH; i++) h_dOutput[i] = 1.0f;

    // Allocate memory on the device
    float* d_input, *d_filters, *d_output, *d_dOutput, *d_dInput;
    hipMalloc(&d_input, INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float));
    hipMalloc(&d_filters, NUM_FILTERS * FILTER_SIZE * FILTER_SIZE * sizeof(float));
    hipMalloc(&d_output, NUM_FILTERS * OUTPUT_HEIGHT * OUTPUT_WIDTH * sizeof(float));
    hipMalloc(&d_dOutput, NUM_FILTERS * OUTPUT_HEIGHT * OUTPUT_WIDTH * sizeof(float));
    hipMalloc(&d_dInput, INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_input, h_input, INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_filters, h_filters, NUM_FILTERS * FILTER_SIZE * FILTER_SIZE * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_dOutput, h_dOutput, NUM_FILTERS * OUTPUT_HEIGHT * OUTPUT_WIDTH * sizeof(float), hipMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((OUTPUT_WIDTH + blockSize.x - 1) / blockSize.x, (OUTPUT_HEIGHT + blockSize.y - 1) / blockSize.y, NUM_FILTERS);

    // Perform forward convolution
    convKernel<<<gridSize, blockSize>>>(d_input, d_filters, d_output, INPUT_HEIGHT, INPUT_WIDTH, FILTER_SIZE, NUM_FILTERS);
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in forward convolution: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Perform backpropagation
    convBackpropKernel<<<gridSize, blockSize>>>(d_dOutput, d_filters, d_dInput, INPUT_HEIGHT, INPUT_WIDTH, FILTER_SIZE, NUM_FILTERS);
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "CUDA error in backpropagation: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    hipMemcpy(h_output, d_output, NUM_FILTERS * OUTPUT_HEIGHT * OUTPUT_WIDTH * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_dInput, d_dInput, INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_input);
    hipFree(d_filters);
    hipFree(d_output);
    hipFree(d_dOutput);
    hipFree(d_dInput);

    // Free host memory
    delete[] h_input;
    delete[] h_filters;
    delete[] h_output;
    delete[] h_dOutput;
    delete[] h_dInput;

    return 0;
}