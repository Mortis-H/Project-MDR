#include<stdio.h>
#include <bits/stdc++.h>
#include<hip/hip_runtime.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

using namespace std;


void get_adj_matrix(float* graph, int n, float d, FILE *inputFilePtr ){

    if ( inputFilePtr == NULL )  {
        printf( "input.txt file failed to open." );
        return ;
    }

    int m, indexing;
    
    fscanf(inputFilePtr, "%d", &m);
    fscanf(inputFilePtr, "%d", &indexing);

    
    for(int i = 0; i< n ; i++){
    
        for(int j = 0; j< n; ++j){
            graph[i* n + j] = (1 - d)/float(n);
        }
    }

    while(m--){
        int source, destin;
        fscanf(inputFilePtr, "%d", &source);
        fscanf(inputFilePtr, "%d", &destin);
        if (indexing == 0){
            graph[destin* n + source] += d* 1.0  ;
        }
        else{
            graph[(destin - 1)* n + source - 1] += d* 1.0;
        }
    }
}

__global__ void manage_adj_matrix(float* gpu_graph, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n){
        float sum = 0.0;

        for (int i = 0; i< n; ++i){
            sum += gpu_graph[i* n + id];
        }

        for (int i = 0; i < n; ++i){
            if (sum != 0.0){
                gpu_graph[i* n + id] /= sum;
            }
            else{
                gpu_graph[i* n + id] = (1/(float)n);
            }
        }
    }
}

__global__ void initialize_rank(float* gpu_r, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        gpu_r[id] = (1/(float)n);
    }
}

__global__ void store_rank(float* gpu_r,float* gpu_r_last, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        gpu_r_last[id] = gpu_r[id];
    }
}

__global__ void matmul(float* gpu_graph, float* gpu_r, float* gpu_r_last, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        float sum = 0.0;

        for (int j = 0; j< n; ++j){
            sum += gpu_r_last[j] * gpu_graph[id* n + j];
        }

        gpu_r[id] = sum;
    }
}

__global__ void rank_diff(float* gpu_r,float* gpu_r_last, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        gpu_r_last[id] = abs(gpu_r_last[id] - gpu_r[id]);
    }
}

__global__ void init_pair_array(pair<float, int>* gpu_r_nodes, float * gpu_r, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < n){
        gpu_r_nodes[id].first = gpu_r[id];
        gpu_r_nodes[id].second = id + 1;
    }
}


void power_method(float *graph, float *r, int n, int nblocks, int BLOCKSIZE, int max_iter = 1000, float eps = 0.000001 ){
   
    float* r_last = (float*) malloc(n * sizeof(float));
    
    float* gpu_graph;
    hipMalloc(&gpu_graph, sizeof(float)*n*n);
    hipMemcpy(gpu_graph, graph, sizeof(float)*n*n, hipMemcpyHostToDevice);

    float* gpu_r;
    hipMalloc(&gpu_r, sizeof(float)*n);
    //cudaMemcpy(gpu_r, r, sizeof(float)*n, cudaMemcpyHostToDevice);

    float* gpu_r_last;
    hipMalloc(&gpu_r_last, sizeof(float)*n);
    //cudaMemcpy(gpu_r_last, r_last, sizeof(float)*n, cudaMemcpyHostToDevice);



    initialize_rank<<<nblocks, BLOCKSIZE>>>(gpu_r, n);
    hipDeviceSynchronize();



    while(max_iter--){

        store_rank<<<nblocks, BLOCKSIZE>>>(gpu_r, gpu_r_last, n);
        hipDeviceSynchronize();

        matmul<<<nblocks, BLOCKSIZE>>>(gpu_graph, gpu_r, gpu_r_last, n);
        hipDeviceSynchronize();
        
        rank_diff<<<nblocks, BLOCKSIZE>>>(gpu_r, gpu_r_last, n);
        hipDeviceSynchronize();

        hipMemcpy(r_last, gpu_r_last, n* sizeof(float), hipMemcpyDeviceToHost);
        float result = thrust::reduce( r_last, r_last + n);

        if(result < eps){
            hipMemcpy(r, gpu_r, n* sizeof(float), hipMemcpyDeviceToHost);
            return;
        }
    }
    hipMemcpy(r, gpu_r, n* sizeof(float), hipMemcpyDeviceToHost);
    return;
}

void top_nodes(float* r, int n, int nblocks, int BLOCKSIZE, int count = 10){

    pair<float, int> *r_nodes = (pair<float, int> *) malloc ( n * sizeof (pair<float, int>) );
    pair<float, int> *gpu_r_nodes;

    hipMalloc(&gpu_r_nodes, n * sizeof (pair<float, int>));

    float* gpu_r;
    hipMalloc(&gpu_r, sizeof(float)*n);
    hipMemcpy(gpu_r, r, sizeof(float)*n, hipMemcpyHostToDevice);

    init_pair_array<<<nblocks, BLOCKSIZE>>>(gpu_r_nodes, gpu_r, n);

    hipMemcpy(r_nodes, gpu_r_nodes, n * sizeof (pair<float, int>), hipMemcpyDeviceToHost);

    thrust::sort(thrust::host, r_nodes, r_nodes + n);

    int rank =1;
    while(rank <= count){
        printf("Rank %d Node is %d\n", rank, r_nodes[n - rank].second);
        rank++;
    }
}

int main(int argc, char** argv){

    clock_t start, end;

    FILE *inputFilePtr;

    char * inputfile = argv[1];

    char * bsize = argv[2];
    int BLOCKSIZE = atoi(bsize);

    inputFilePtr = fopen(inputfile, "r");

    int n; 

    fscanf(inputFilePtr, "%d", &n);
    int nblocks = ceil(float(n) / BLOCKSIZE);

    float* graph = (float*)malloc(n*n*sizeof(float));
    float* r = (float*) malloc(n * sizeof(float));

    float d = 0.85;

    get_adj_matrix(graph, n, d, inputFilePtr);

    float* gpu_graph;
    hipMalloc(&gpu_graph, sizeof(float)*n*n);
    hipMemcpy(gpu_graph, graph, sizeof(float)*n*n, hipMemcpyHostToDevice);
    
    start = clock();

    manage_adj_matrix<<<nblocks, BLOCKSIZE>>>(gpu_graph, n);
    hipMemcpy(graph, gpu_graph, sizeof(float)*n*n, hipMemcpyDeviceToHost);

    power_method(graph, r, n, nblocks, BLOCKSIZE );

    top_nodes(r, n, nblocks, BLOCKSIZE);

    end = clock();

    printf("Time taken :%f for parallel implementation with %d nodes.\n", float(end - start), n);
    return 0;
}