#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <math.h>

#define SHIFT       50
#define SCALE       5

#define ARRAY_SIZE  1000
#define BLOCK_SIZE  512

#define CUDA_CALL(ans) { GpuAssert((ans), __FILE__, __LINE__); }
inline void GpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess) 
   {
      printf("GPU assert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void normal_init_kernel(hiprandState *state, float seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets different seed, a different sequence number, no offset */
    hiprand_init(seed, idx, 0, &state[idx]);
}

__global__ void normal_generate_kernel(hiprandState *state, float *result)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    hiprandState localState = state[idx];                            /* Copy state to local memory for efficiency */
    result[idx] = (hiprand_normal(&localState) * SCALE)+SHIFT;       /* Generate pseudo-random uniforms */
    state[idx] = localState;                                        /* Copy state back to global memory */
}

void normal_generator(float seed){
    hiprandState *dev_states;
    float *dev_array, *hst_array;

    hst_array = (float *)malloc(ARRAY_SIZE * sizeof(float));
    CUDA_CALL(hipMalloc((void **)&dev_array, ARRAY_SIZE *sizeof(float)));
    CUDA_CALL(hipMalloc((void **)&dev_states, ARRAY_SIZE * sizeof(hiprandState)));

    int grid_size = ARRAY_SIZE / BLOCK_SIZE + 1;
    normal_init_kernel<<<grid_size, BLOCK_SIZE>>>(dev_states, seed);
    normal_generate_kernel<<<grid_size, BLOCK_SIZE>>>(dev_states, dev_array);
    
    /* Copy device memory to host */
    CUDA_CALL(hipMemcpy(hst_array, dev_array, ARRAY_SIZE * sizeof(int), hipMemcpyDeviceToHost));

    /* Show result */
    int p[6*SCALE]={};
    for (int i=0; i<ARRAY_SIZE; i++) {
        if(hst_array[i] >= SHIFT-3*SCALE && SHIFT-SCALE < SHIFT+3*SCALE){
            p[(int)hst_array[i] - (SHIFT-3*SCALE)] ++;
        }
    }
    for (int i=SHIFT-3*SCALE; i<SHIFT+3*SCALE; ++i) {
        std::cout << i << "-" << (i+1) << ":";
        std::cout << "  " << std::string(p[i - (SHIFT-3*SCALE)],'*') << std::endl;
    }
    
    float total = 0;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        total += hst_array[i];
    }
    printf("seed: %f, Results mean = %f\n", seed,(total/(1.0*ARRAY_SIZE)));

    /* Cleanup */
    CUDA_CALL(hipFree(dev_array));
    CUDA_CALL(hipFree(dev_states));
    free(hst_array);

    hipDeviceSynchronize();
}


int main(int argc, char *argv[])
{
    int device;
    struct hipDeviceProp_t properties;

    CUDA_CALL(hipGetDevice(&device));
    CUDA_CALL(hipGetDeviceProperties(&properties,device));

    for(int i=0; i< 10; i++){
        normal_generator(i);
    }

    return 0;
}