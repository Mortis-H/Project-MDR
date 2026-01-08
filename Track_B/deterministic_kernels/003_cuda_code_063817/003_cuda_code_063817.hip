#include <iostream>
#include <hip/hip_runtime.h>

// Define the convolution parameters
#define FILTER_WIDTH 3
#define FILTER_HEIGHT 3
#define INPUT_CHANNELS 1
#define OUTPUT_CHANNELS 1
#define INPUT_WIDTH 10
#define INPUT_HEIGHT 10
#define OUTPUT_WIDTH (INPUT_WIDTH - FILTER_WIDTH + 1)
#define OUTPUT_HEIGHT (INPUT_HEIGHT - FILTER_HEIGHT + 1)

// Kernel for forward pass (convolution)
__global__ void forwardConvolution(const float* input, const float* filter, float* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < OUTPUT_WIDTH && y < OUTPUT_HEIGHT) {
        float sum = 0.0f;
        for (int c = 0; c < INPUT_CHANNELS; ++c) {
            for (int fy = 0; fy < FILTER_HEIGHT; ++fy) {
                for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
                    int ix = x + fx;
                    int iy = y + fy;
                    sum += input[(iy * INPUT_WIDTH + ix) * INPUT_CHANNELS + c] * filter[(fy * FILTER_WIDTH + fx) * INPUT_CHANNELS + c];
                }
            }
        }
        output[(y * OUTPUT_WIDTH + x) * OUTPUT_CHANNELS] = sum;
    }
}

// Kernel for backpropagation of weights
__global__ void backpropWeights(const float* input, const float* outputGradient, float* weightGradient) {
    int fx = blockIdx.x * blockDim.x + threadIdx.x;
    int fy = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (fx < FILTER_WIDTH && fy < FILTER_HEIGHT) {
        float sum = 0.0f;
        for (int y = 0; y < OUTPUT_HEIGHT; ++y) {
            for (int x = 0; x < OUTPUT_WIDTH; ++x) {
                int ix = x + fx;
                int iy = y + fy;
                sum += input[(iy * INPUT_WIDTH + ix) * INPUT_CHANNELS + c] * outputGradient[(y * OUTPUT_WIDTH + x) * OUTPUT_CHANNELS];
            }
        }
        atomicAdd(&weightGradient[(fy * FILTER_WIDTH + fx) * INPUT_CHANNELS + c], sum);
    }
}

// Kernel for backpropagation of input
__global__ void backpropInput(const float* outputGradient, const float* filter, float* inputGradient) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x < INPUT_WIDTH && y < INPUT_HEIGHT) {
        float sum = 0.0f;
        for (int fy = 0; fy < FILTER_HEIGHT; ++fy) {
            for (int fx = 0; fx < FILTER_WIDTH; ++fx) {
                int ox = x - fx;
                int oy = y - fy;
                if (ox >= 0 && oy >= 0 && ox < OUTPUT_WIDTH && oy < OUTPUT_HEIGHT) {
                    sum += outputGradient[(oy * OUTPUT_WIDTH + ox) * OUTPUT_CHANNELS] * filter[(fy * FILTER_WIDTH + fx) * INPUT_CHANNELS + c];
                }
            }
        }
        atomicAdd(&inputGradient[(y * INPUT_WIDTH + x) * INPUT_CHANNELS + c], sum);
    }
}

int main() {
    // Allocate and initialize host memory
    float* h_input = new float[INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS];
    float* h_filter = new float[FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNELS];
    float* h_output = new float[OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS];
    float* h_outputGradient = new float[OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS];
    float* h_weightGradient = new float[FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNELS];
    float* h_inputGradient = new float[INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS];

    // Initialize input, filter, and output gradient with some values
    for (int i = 0; i < INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS; ++i) h_input[i] = 1.0f;
    for (int i = 0; i < FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNELS; ++i) h_filter[i] = 0.5f;
    for (int i = 0; i < OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS; ++i) h_outputGradient[i] = 1.0f;
    for (int i = 0; i < FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNELS; ++i) h_weightGradient[i] = 0.0f;
    for (int i = 0; i < INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS; ++i) h_inputGradient[i] = 0.0f;

    // Allocate device memory
    float* d_input, *d_filter, *d_output, *d_outputGradient, *d_weightGradient, *d_inputGradient;
    hipMalloc(&d_input, INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS * sizeof(float));
    hipMalloc(&d_filter, FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNELS * sizeof(float));
    hipMalloc(&d_output, OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS * sizeof(float));
    hipMalloc(&d_outputGradient, OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS * sizeof(float));
    hipMalloc(&d_weightGradient, FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNELS * sizeof(float));
    hipMalloc(&d_inputGradient, INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS * sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_input, h_input, INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_filter, h_filter, FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNELS * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_outputGradient, h_outputGradient, OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS * sizeof(float), hipMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((OUTPUT_WIDTH + blockSize.x - 1) / blockSize.x, (OUTPUT_HEIGHT + blockSize.y - 1) / blockSize.y);

    // Launch forward convolution kernel
    forwardConvolution<<<gridSize, blockSize>>>(d_input, d_filter, d_output);
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Forward convolution kernel failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch backpropagation of weights kernel
    dim3 weightGradientBlockSize(8, 8);
    dim3 weightGradientGridSize((FILTER_WIDTH + weightGradientBlockSize.x - 1) / weightGradientBlockSize.x, 
                                (FILTER_HEIGHT + weightGradientBlockSize.y - 1) / weightGradientBlockSize.y, 
                                INPUT_CHANNELS);
    backpropWeights<<<weightGradientGridSize, weightGradientBlockSize>>>(d_input, d_outputGradient, d_weightGradient);
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Backpropagation of weights kernel failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Launch backpropagation of input kernel
    dim3 inputGradientBlockSize(16, 16);
    dim3 inputGradientGridSize((INPUT_WIDTH + inputGradientBlockSize.x - 1) / inputGradientBlockSize.x, 
                              (INPUT_HEIGHT + inputGradientBlockSize.y - 1) / inputGradientBlockSize.y, 
                              INPUT_CHANNELS);
    backpropInput<<<inputGradientGridSize, inputGradientBlockSize>>>(d_outputGradient, d_filter, d_inputGradient);
    hipDeviceSynchronize();
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Backpropagation of input kernel failed: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    hipMemcpy(h_output, d_output, OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_weightGradient, d_weightGradient, FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNELS * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_inputGradient, d_inputGradient, INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_input);
    hipFree(d_filter);
    hipFree(d_output);
    hipFree(d_outputGradient);
    hipFree(d_weightGradient);
    hipFree(d_inputGradient);

    // Free host memory
    delete[] h_input;
    delete[] h_filter;
    delete[] h_output;
    delete[] h_outputGradient;
    delete[] h_weightGradient;
    delete[] h_inputGradient;

    return 0;
}