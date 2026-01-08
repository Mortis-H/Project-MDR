#include <iostream>
#include <hip/hip_runtime.h>


#define BLOCK_SIZE 256
#define NUM_TOKENS 16384
#define HEADS 8
#define EMBED_DIM 512

__global__ void computeQKV(float* Q, float* K, float* V, const float* input, int embed_dim, int heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_TOKENS * embed_dim) {
        int token = idx / embed_dim;
        int dim = idx % embed_dim;
        int head_dim = embed_dim / heads;
        int head = dim / head_dim;
        int head_dim_idx = dim % head_dim;

        Q[token * embed_dim + dim] = input[token * embed_dim + dim];
        K[token * embed_dim + dim] = input[token * embed_dim + dim];
        V[token * embed_dim + dim] = input[token * embed_dim + dim];
    }
}

__global__ void computeAttention(float* output, const float* Q, const float* K, const float* V, int embed_dim, int heads) {
    extern __shared__ float shared_mem[];
    float* Q_shared = shared_mem;
    float* K_shared = shared_mem + BLOCK_SIZE * embed_dim / heads;
    float* V_shared = shared_mem + 2 * BLOCK_SIZE * embed_dim / heads;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token = idx / embed_dim;
    int dim = idx % embed_dim;
    int head_dim = embed_dim / heads;
    int head = dim / head_dim;
    int head_dim_idx = dim % head_dim;

    if (idx < NUM_TOKENS * embed_dim) {
        Q_shared[threadIdx.x * head_dim + head_dim_idx] = Q[token * embed_dim + dim];
        K_shared[threadIdx.x * head_dim + head_dim_idx] = K[token * embed_dim + dim];
        V_shared[threadIdx.x * head_dim + head_dim_idx] = V[token * embed_dim + dim];
        __syncthreads();

        float sum = 0.0f;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            sum += Q_shared[threadIdx.x * head_dim + head_dim_idx] * K_shared[j * head_dim + head_dim_idx];
        }
        __syncthreads();

        float softmax = expf(sum) / (NUM_TOKENS * 1.0f);
        output[token * embed_dim + dim] = softmax * V_shared[threadIdx.x * head_dim + head_dim_idx];
    }
}

void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    float* d_input, *d_Q, *d_K, *d_V, *d_output;
    int embed_dim = EMBED_DIM;
    int size = NUM_TOKENS * embed_dim * sizeof(float);

    checkCudaError(hipMalloc(&d_input, size), "Failed to allocate d_input");
    checkCudaError(hipMalloc(&d_Q, size), "Failed to allocate d_Q");
    checkCudaError(hipMalloc(&d_K, size), "Failed to allocate d_K");
    checkCudaError(hipMalloc(&d_V, size), "Failed to allocate d_V");
    checkCudaError(hipMalloc(&d_output, size), "Failed to allocate d_output");

    // Initialize input data on host
    float* h_input = new float[NUM_TOKENS * embed_dim];
    for (int i = 0; i < NUM_TOKENS * embed_dim; ++i) {
        h_input[i] = static_cast<float>(i) / 1000.0f;
    }

    // Copy input data to device
    checkCudaError(hipMemcpy(d_input, h_input, size, hipMemcpyHostToDevice), "Failed to copy input to device");

    // Launch kernel to compute Q, K, V
    dim3 grid((NUM_TOKENS * embed_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    computeQKV<<<grid, block>>>(d_Q, d_K, d_V, d_input, embed_dim, HEADS);
    checkCudaError(hipGetLastError(), "Failed to launch computeQKV kernel");
    checkCudaError(hipDeviceSynchronize(), "Failed to synchronize after computeQKV kernel");

    // Launch kernel to compute attention
    computeAttention<<<grid, block, 3 * BLOCK_SIZE * embed_dim / HEADS * sizeof(float)>>>(d_output, d_Q, d_K, d_V, embed_dim, HEADS);
    checkCudaError(hipGetLastError(), "Failed to launch computeAttention kernel");
    checkCudaError(hipDeviceSynchronize(), "Failed to synchronize after computeAttention kernel");

    // Copy output data to host
    float* h_output = new float[NUM_TOKENS * embed_dim];
    checkCudaError(hipMemcpy(h_output, d_output, size, hipMemcpyDeviceToHost), "Failed to copy output to host");

    // Free device memory
    checkCudaError(hipFree(d_input), "Failed to free d_input");
    checkCudaError(hipFree(d_Q), "Failed to free d_Q");
    checkCudaError(hipFree(d_K), "Failed to free d_K");
    checkCudaError(hipFree(d_V), "Failed to free d_V");
    checkCudaError(hipFree(d_output), "Failed to free d_output");

    // Free host memory
    delete[] h_input;
    delete[] h_output;

    std::cout << "CUDA attention mechanism computation completed successfully." << std::endl;

    return 0;
}