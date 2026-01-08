#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define NUM_TOKENS 128
#define HEADS 8
#define EMBED_DIM 128
#define EMBED_PER_HEAD (EMBED_DIM / HEADS)

__global__ void compute_qkv(float* input, float* q, float* k, float* v, int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = tid / (NUM_TOKENS * EMBED_DIM);
    int token = (tid / EMBED_DIM) % NUM_TOKENS;
    int dim = tid % EMBED_DIM;

    if (batch < batch_size) {
        int q_index = batch * NUM_TOKENS * EMBED_DIM + token * EMBED_DIM + dim;
        int k_index = batch * NUM_TOKENS * EMBED_DIM + token * EMBED_DIM + dim;
        int v_index = batch * NUM_TOKENS * EMBED_DIM + token * EMBED_DIM + dim;
        int input_index = batch * NUM_TOKENS * EMBED_DIM + token * EMBED_DIM + dim;

        q[q_index] = input[input_index]; // Simplified Q computation
        k[k_index] = input[input_index]; // Simplified K computation
        v[v_index] = input[input_index]; // Simplified V computation
    }
}

__global__ void compute_attention_scores(float* q, float* k, float* scores, int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = tid / (NUM_TOKENS * NUM_TOKENS);
    int token_i = (tid / NUM_TOKENS) % NUM_TOKENS;
    int token_j = tid % NUM_TOKENS;

    if (batch < batch_size) {
        float score = 0.0f;
        for (int dim = 0; dim < EMBED_DIM; ++dim) {
            int q_index = batch * NUM_TOKENS * EMBED_DIM + token_i * EMBED_DIM + dim;
            int k_index = batch * NUM_TOKENS * EMBED_DIM + token_j * EMBED_DIM + dim;
            score += q[q_index] * k[k_index];
        }
        scores[batch * NUM_TOKENS * NUM_TOKENS + token_i * NUM_TOKENS + token_j] = score / sqrtf(EMBED_DIM);
    }
}

__global__ void softmax(float* scores, int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = tid / NUM_TOKENS;
    int token_i = tid % NUM_TOKENS;

    if (batch < batch_size) {
        float max_score = -1e9f;
        for (int token_j = 0; token_j < NUM_TOKENS; ++token_j) {
            int index = batch * NUM_TOKENS * NUM_TOKENS + token_i * NUM_TOKENS + token_j;
            max_score = fmaxf(max_score, scores[index]);
        }

        float sum_exp = 0.0f;
        for (int token_j = 0; token_j < NUM_TOKENS; ++token_j) {
            int index = batch * NUM_TOKENS * NUM_TOKENS + token_i * NUM_TOKENS + token_j;
            scores[index] = expf(scores[index] - max_score);
            sum_exp += scores[index];
        }

        for (int token_j = 0; token_j < NUM_TOKENS; ++token_j) {
            int index = batch * NUM_TOKENS * NUM_TOKENS + token_i * NUM_TOKENS + token_j;
            scores[index] /= sum_exp;
        }
    }
}

__global__ void compute_weighted_sum(float* scores, float* v, float* output, int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = tid / (NUM_TOKENS * EMBED_DIM);
    int token_i = (tid / EMBED_DIM) % NUM_TOKENS;
    int dim = tid % EMBED_DIM;

    if (batch < batch_size) {
        float sum = 0.0f;
        for (int token_j = 0; token_j < NUM_TOKENS; ++token_j) {
            int score_index = batch * NUM_TOKENS * NUM_TOKENS + token_i * NUM_TOKENS + token_j;
            int v_index = batch * NUM_TOKENS * EMBED_DIM + token_j * EMBED_DIM + dim;
            sum += scores[score_index] * v[v_index];
        }
        output[batch * NUM_TOKENS * EMBED_DIM + token_i * EMBED_DIM + dim] = sum;
    }
}

void checkCudaError(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        std::cerr << "CUDA error: " << msg << " (" << hipGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int batch_size = 1;
    size_t input_size = batch_size * NUM_TOKENS * EMBED_DIM * sizeof(float);
    size_t qkv_size = input_size;
    size_t scores_size = batch_size * NUM_TOKENS * NUM_TOKENS * sizeof(float);
    size_t output_size = input_size;

    float* d_input, *d_q, *d_k, *d_v, *d_scores, *d_output;
    checkCudaError(hipMalloc(&d_input, input_size), "Failed to allocate d_input");
    checkCudaError(hipMalloc(&d_q, qkv_size), "Failed to allocate d_q");
    checkCudaError(hipMalloc(&d_k, qkv_size), "Failed to allocate d_k");
    checkCudaError(hipMalloc(&d_v, qkv_size), "Failed to allocate d_v");
    checkCudaError(hipMalloc(&d_scores, scores_size), "Failed to allocate d_scores");
    checkCudaError(hipMalloc(&d_output, output_size), "Failed to allocate d_output");

    // Initialize input data
    float* h_input = new float[batch_size * NUM_TOKENS * EMBED_DIM];
    for (int i = 0; i < batch_size * NUM_TOKENS * EMBED_DIM; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    checkCudaError(hipMemcpy(d_input, h_input, input_size, hipMemcpyHostToDevice), "Failed to copy input to device");

    dim3 blockSize(256);
    dim3 numBlocks((batch_size * NUM_TOKENS * EMBED_DIM + blockSize.x - 1) / blockSize.x);

    compute_qkv<<<numBlocks, blockSize>>>(d_input, d_q, d_k, d_v, batch_size);
    checkCudaError(hipGetLastError(), "compute_qkv kernel launch failed");

    numBlocks = dim3((batch_size * NUM_TOKENS * NUM_TOKENS + blockSize.x - 1) / blockSize.x);
    compute_attention_scores<<<numBlocks, blockSize>>>(d_q, d_k, d_scores, batch_size);
    checkCudaError(hipGetLastError(), "compute_attention_scores kernel launch failed");

    numBlocks = dim3((batch_size * NUM_TOKENS + blockSize.x - 1) / blockSize.x);
    softmax<<<numBlocks, blockSize>>>(d_scores, batch_size);
    checkCudaError(hipGetLastError(), "softmax kernel launch failed");

    numBlocks = dim3((batch_size * NUM_TOKENS * EMBED_DIM + blockSize.x - 1) / blockSize.x);
    compute_weighted_sum<<<numBlocks, blockSize>>>(d_scores, d_v, d_output, batch_size);
    checkCudaError(hipGetLastError(), "compute_weighted_sum kernel launch failed");

    float* h_output = new float[batch_size * NUM_TOKENS * EMBED_DIM];
    checkCudaError(hipMemcpy(h_output, d_output, output_size, hipMemcpyDeviceToHost), "Failed to copy output to host");

    // Free device memory
    hipFree(d_input);
    hipFree(d_q);
    hipFree(d_k);
    hipFree(d_v);
    hipFree(d_scores);
    hipFree(d_output);

    // Free host memory
    delete[] h_input;
    delete[] h_output;

    return 0;
}