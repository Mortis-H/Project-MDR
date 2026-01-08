#include <iostream>
#include <hip/hip_runtime.h>

#include <hip/hip_fp16.h>

#define NUM_TOKENS 256
#define HEADS 8
#define EMBED_DIM 256
#define DIM_PER_HEAD (EMBED_DIM / HEADS)

__global__ void compute_query_key_value(float* input, float* query, float* key, float* value, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_TOKENS * embed_dim) {
        int token = idx / embed_dim;
        int dim = idx % embed_dim;
        int head = dim / DIM_PER_HEAD;
        int dim_in_head = dim % DIM_PER_HEAD;

        query[token * HEADS * DIM_PER_HEAD + head * DIM_PER_HEAD + dim_in_head] = input[idx];
        key[token * HEADS * DIM_PER_HEAD + head * DIM_PER_HEAD + dim_in_head] = input[idx];
        value[token * HEADS * DIM_PER_HEAD + head * DIM_PER_HEAD + dim_in_head] = input[idx];
    }
}

__global__ void compute_attention_scores(float* query, float* key, float* scores, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_TOKENS * NUM_TOKENS * HEADS) {
        int token_q = idx / (NUM_TOKENS * HEADS);
        int token_k = (idx / HEADS) % NUM_TOKENS;
        int head = idx % HEADS;
        float score = 0.0f;

        for (int d = 0; d < DIM_PER_HEAD; ++d) {
            score += query[token_q * HEADS * DIM_PER_HEAD + head * DIM_PER_HEAD + d] *
                     key[token_k * HEADS * DIM_PER_HEAD + head * DIM_PER_HEAD + d];
        }
        scores[idx] = score / sqrtf(DIM_PER_HEAD);
    }
}

__global__ void softmax(float* scores, float* softmax_output, int num_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tokens * num_tokens * HEADS) {
        int row = idx / (num_tokens * HEADS);
        int col = (idx / HEADS) % num_tokens;
        int head = idx % HEADS;

        float max_val = -1e9f;
        for (int i = 0; i < num_tokens; ++i) {
            max_val = fmaxf(max_val, scores[row * num_tokens * HEADS + i * HEADS + head]);
        }

        float sum = 0.0f;
        for (int i = 0; i < num_tokens; ++i) {
            sum += expf(scores[row * num_tokens * HEADS + i * HEADS + head] - max_val);
        }

        softmax_output[idx] = expf(scores[idx] - max_val) / sum;
    }
}

__global__ void compute_context_values(float* softmax_output, float* value, float* context, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_TOKENS * embed_dim) {
        int token = idx / embed_dim;
        int dim = idx % embed_dim;
        int head = dim / DIM_PER_HEAD;
        int dim_in_head = dim % DIM_PER_HEAD;

        float value_sum = 0.0f;
        for (int k = 0; k < NUM_TOKENS; ++k) {
            value_sum += softmax_output[token * NUM_TOKENS * HEADS + k * HEADS + head] *
                         value[k * HEADS * DIM_PER_HEAD + head * DIM_PER_HEAD + dim_in_head];
        }
        context[token * HEADS * DIM_PER_HEAD + head * DIM_PER_HEAD + dim_in_head] = value_sum;
    }
}

__global__ void finalize_output(float* context, float* output, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_TOKENS * embed_dim) {
        output[idx] = context[idx];
    }
}

void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    float* d_input, *d_query, *d_key, *d_value, *d_scores, *d_softmax_output, *d_context, *d_output;
    float* h_input = new float[NUM_TOKENS * EMBED_DIM];
    float* h_output = new float[NUM_TOKENS * EMBED_DIM];

    // Initialize input with random values
    for (int i = 0; i < NUM_TOKENS * EMBED_DIM; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    checkCudaError(hipMalloc(&d_input, NUM_TOKENS * EMBED_DIM * sizeof(float)), "hipMalloc d_input");
    checkCudaError(hipMalloc(&d_query, NUM_TOKENS * HEADS * DIM_PER_HEAD * sizeof(float)), "hipMalloc d_query");
    checkCudaError(hipMalloc(&d_key, NUM_TOKENS * HEADS * DIM_PER_HEAD * sizeof(float)), "hipMalloc d_key");
    checkCudaError(hipMalloc(&d_value, NUM_TOKENS * HEADS * DIM_PER_HEAD * sizeof(float)), "hipMalloc d_value");
    checkCudaError(hipMalloc(&d_scores, NUM_TOKENS * NUM_TOKENS * HEADS * sizeof(float)), "hipMalloc d_scores");
    checkCudaError(hipMalloc(&d_softmax_output, NUM_TOKENS * NUM_TOKENS * HEADS * sizeof(float)), "hipMalloc d_softmax_output");
    checkCudaError(hipMalloc(&d_context, NUM_TOKENS * HEADS * DIM_PER_HEAD * sizeof(float)), "hipMalloc d_context");
    checkCudaError(hipMalloc(&d_output, NUM_TOKENS * EMBED_DIM * sizeof(float)), "hipMalloc d_output");

    // Copy input to device
    checkCudaError(hipMemcpy(d_input, h_input, NUM_TOKENS * EMBED_DIM * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy d_input");

    // Launch kernels
    int blockSize = 256;
    int numBlocks = (NUM_TOKENS * EMBED_DIM + blockSize - 1) / blockSize;
    compute_query_key_value<<<numBlocks, blockSize>>>(d_input, d_query, d_key, d_value, EMBED_DIM);
    checkCudaError(hipGetLastError(), "compute_query_key_value");

    numBlocks = (NUM_TOKENS * NUM_TOKENS * HEADS + blockSize - 1) / blockSize;
    compute_attention_scores<<<numBlocks, blockSize>>>(d_query, d_key, d_scores, EMBED_DIM);
    checkCudaError(hipGetLastError(), "compute_attention_scores");

    softmax<<<numBlocks, blockSize>>>(d_scores, d_softmax_output, NUM_TOKENS);
    checkCudaError(hipGetLastError(), "softmax");

    compute_context_values<<<numBlocks, blockSize>>>(d_softmax_output, d_value, d_context, EMBED_DIM);
    checkCudaError(hipGetLastError(), "compute_context_values");

    numBlocks = (NUM_TOKENS * EMBED_DIM + blockSize - 1) / blockSize;
    finalize_output<<<numBlocks, blockSize>>>(d_context, d_output, EMBED_DIM);
    checkCudaError(hipGetLastError(), "finalize_output");

    // Copy output back to host
    checkCudaError(hipMemcpy(h_output, d_output, NUM_TOKENS * EMBED_DIM * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy d_output");

    // Free device memory
    hipFree(d_input);
    hipFree(d_query);
    hipFree(d_key);
    hipFree(d_value);
    hipFree(d_scores);
    hipFree(d_softmax_output);
    hipFree(d_context);
    hipFree(d_output);

    // Free host memory
    delete[] h_input;
    delete[] h_output;

    std::cout << "Attention mechanism computed successfully." << std::endl;

    return 0;
}