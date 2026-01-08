#include <iostream>
#include <hip/hip_runtime.h>

#define NUM_TOKENS 1024
#define HEADS 8
#define EMBED_DIM 64
#define HIDDEN_DIM 256

__global__ void computeQueryKeyMatrices(float* Q, float* K, const float* X, int numTokens, int embedDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTokens * embedDim) {
        int token = idx / embedDim;
        int dim = idx % embedDim;
        Q[token * embedDim + dim] = X[token * embedDim + dim]; // Simple copy for demonstration
        K[token * embedDim + dim] = X[token * embedDim + dim]; // Simple copy for demonstration
    }
}

__global__ void computeAttentionScores(float* scores, const float* Q, const float* K, int numTokens, int embedDim) {
    int row = blockIdx.x;
    int col = blockIdx.y;
    if (row < numTokens && col < numTokens) {
        float score = 0.0f;
        for (int d = 0; d < embedDim; ++d) {
            score += Q[row * embedDim + d] * K[col * embedDim + d];
        }
        scores[row * numTokens + col] = score / sqrtf(static_cast<float>(embedDim));
    }
}

__global__ void applySoftmax(float* softmaxScores, const float* scores, int numTokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTokens) {
        float sum = 0.0f;
        for (int j = 0; j < numTokens; ++j) {
            sum += expf(scores[idx * numTokens + j]);
        }
        for (int j = 0; j < numTokens; ++j) {
            softmaxScores[idx * numTokens + j] = expf(scores[idx * numTokens + j]) / sum;
        }
    }
}

__global__ void computeContextValues(float* context, const float* softmaxScores, const float* V, int numTokens, int embedDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTokens * embedDim) {
        int token = idx / embedDim;
        int dim = idx % embedDim;
        float value = 0.0f;
        for (int j = 0; j < numTokens; ++j) {
            value += softmaxScores[token * numTokens + j] * V[j * embedDim + dim];
        }
        context[token * embedDim + dim] = value;
    }
}

int main() {
    float* d_X, *d_Q, *d_K, *d_V, *d_scores, *d_softmaxScores, *d_context;
    float* h_X = new float[NUM_TOKENS * EMBED_DIM];
    float* h_context = new float[NUM_TOKENS * EMBED_DIM];

    // Initialize X with random values
    for (int i = 0; i < NUM_TOKENS * EMBED_DIM; ++i) {
        h_X[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on the device
    hipError_t err = hipMalloc(&d_X, NUM_TOKENS * EMBED_DIM * sizeof(float));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_Q, NUM_TOKENS * EMBED_DIM * sizeof(float));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_K, NUM_TOKENS * EMBED_DIM * sizeof(float));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_V, NUM_TOKENS * EMBED_DIM * sizeof(float));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_scores, NUM_TOKENS * NUM_TOKENS * sizeof(float));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_softmaxScores, NUM_TOKENS * NUM_TOKENS * sizeof(float));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMalloc(&d_context, NUM_TOKENS * EMBED_DIM * sizeof(float));
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    // Copy data to the device
    err = hipMemcpy(d_X, h_X, NUM_TOKENS * EMBED_DIM * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }
    err = hipMemcpy(d_V, h_X, NUM_TOKENS * EMBED_DIM * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    // Launch kernels
    int blockSize = 256;
    int numBlocks = (NUM_TOKENS * EMBED_DIM + blockSize - 1) / blockSize;
    computeQueryKeyMatrices<<<numBlocks, blockSize>>>(d_Q, d_K, d_X, NUM_TOKENS, EMBED_DIM);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    dim3 grid(NUM_TOKENS, NUM_TOKENS);
    computeAttentionScores<<<grid, 1>>>(d_scores, d_Q, d_K, NUM_TOKENS, EMBED_DIM);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    numBlocks = (NUM_TOKENS + blockSize - 1) / blockSize;
    applySoftmax<<<numBlocks, blockSize>>>(d_softmaxScores, d_scores, NUM_TOKENS);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    computeContextValues<<<numBlocks, blockSize>>>(d_context, d_softmaxScores, d_V, NUM_TOKENS, EMBED_DIM);
    err = hipGetLastError();
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    // Copy result back to host
    err = hipMemcpy(h_context, d_context, NUM_TOKENS * EMBED_DIM * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) { std::cerr << "CUDA error: " << hipGetErrorString(err) << std::endl; return -1; }

    // Free device memory
    hipFree(d_X);
    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_scores);
    hipFree(d_softmaxScores);
    hipFree(d_context);

    // Free host memory
    delete[] h_X;
    delete[] h_context;

    return 0;
}