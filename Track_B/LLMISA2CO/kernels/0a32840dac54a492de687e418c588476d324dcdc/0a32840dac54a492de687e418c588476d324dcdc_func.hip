__global__ void matDet(double *d_matA, double *detM){ 
    
	int global_id =  blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ double s_mat[sizeof(double)*64];
    __shared__ double s_detAux[16];
    int offset = (threadIdx.y * blockDim.x + threadIdx.x)*16; 

    if ((threadIdx.y * blockDim.x + threadIdx.x) < 64){
        s_mat[(threadIdx.y * blockDim.x + threadIdx.x)]=d_matA[global_id];
        if(threadIdx.y * blockDim.x + threadIdx.x < 16){
            s_detAux[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }
        __syncthreads();
 
        if(threadIdx.y * blockDim.x + threadIdx.x < 4){
          //  printf("globalId:%d|%d|%d|%d|%d\n",global_id,(threadIdx.y * blockDim.x + threadIdx.x)*4,(threadIdx.y * blockDim.x + threadIdx.x)*4+1,(threadIdx.y * blockDim.x + threadIdx.x)*4+2,(threadIdx.y * blockDim.x + threadIdx.x)*4+3);
            s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4] +=  s_mat[offset] * ( (s_mat[offset+5]*s_mat[offset+10]*s_mat[offset+15])+(s_mat[offset+6]*s_mat[offset+11]*s_mat[offset+13])+(s_mat[offset+7]*s_mat[offset+9]*s_mat[offset+14])   +  (-1*(s_mat[offset+7]*s_mat[offset+10]*s_mat[offset+13]))   + (-1*(s_mat[offset+5]*s_mat[offset+11]*s_mat[offset+14]))  + (-1*(s_mat[offset+6]*s_mat[offset+9]*s_mat[offset+15])) );
    //        __syncthreads();       
           
            s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4+1] +=  (-1*s_mat[offset+1]) * ( (s_mat[offset+4]*s_mat[offset+10]*s_mat[offset+15])+(s_mat[offset+6]*s_mat[offset+11]*s_mat[offset+12])+(s_mat[offset+7]*s_mat[offset+8]*s_mat[offset+14])   +  (-1*(s_mat[offset+7]*s_mat[offset+10]*s_mat[offset+12]))   + (-1*(s_mat[offset+4]*s_mat[offset+11]*s_mat[offset+14]))  + (-1*(s_mat[offset+6]*s_mat[offset+8]*s_mat[offset+15])) );    
     //       __syncthreads();

            s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4+2] +=  s_mat[offset+2] * ( (s_mat[offset+4]*s_mat[offset+9]*s_mat[offset+15])+(s_mat[offset+5]*s_mat[offset+11]*s_mat[offset+12])+(s_mat[offset+7]*s_mat[offset+8]*s_mat[offset+13])   +  (-1*(s_mat[offset+7]*s_mat[offset+9]*s_mat[offset+12]))   + (-1*(s_mat[offset+4]*s_mat[offset+11]*s_mat[offset+13]))  + (-1*(s_mat[offset+5]*s_mat[offset+8]*s_mat[offset+15])) );        
     //       __syncthreads();
      
            s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4+3] +=  (-1*s_mat[offset+3]) * ( (s_mat[offset+4]*s_mat[offset+9]*s_mat[offset+14])+(s_mat[offset+5]*s_mat[offset+10]*s_mat[offset+12])+(s_mat[offset+6]*s_mat[offset+8]*s_mat[offset+13])   +  (-1*(s_mat[offset+6]*s_mat[offset+9]*s_mat[offset+12]))   + (-1*(s_mat[offset+4]*s_mat[offset+10]*s_mat[offset+13]))  + (-1*(s_mat[offset+5]*s_mat[offset+8]*s_mat[offset+14])) );            
       //     __syncthreads();
          
            detM[blockIdx.x*4 + (threadIdx.y * blockDim.x + threadIdx.x)] = s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4] + s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4+1] + s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4+2] + s_detAux[(threadIdx.y * blockDim.x + threadIdx.x)*4+3]; 
  //          __syncthreads();
          
        }
    }
}