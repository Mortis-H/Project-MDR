#include<stdio.h>
#include<stdlib.h>
#include<hip/hip_runtime.h>
#include <sys/time.h>
#include<thrust/sort.h>
#include<math.h>
#define BLOCKSIZE 1024

struct vehicle
{
	float time ; 
	int id;
};
struct cmp {
  __host__ __device__
  bool operator()(const vehicle& o1, const vehicle& o2) {
      if (o1.time == o2.time)
      	return o1.id < o2.id ;
      else
      	return o1.time < o2.time;
  }
};
__global__ void dkernel(int n,int k,int size,float *matrix , int dis , int *speed)
{
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x ;
	if (id < size)
	{
      float time = (float)dis/(float)speed[id];
		  matrix[id] = time*(float)60;
	}
}
__global__ void dkernel1(float *matrix ,vehicle* AT, int i,int n)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x ;
    if(idx < n)
    {
        AT[idx].time=matrix[i*n + idx];
        AT[idx].id = idx ;
    }
}
__global__ void dkernel2(float *matrix, vehicle* AT , float* ET,int i ,int n)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x ;
    if(idx < n)
    {
        AT[idx].time = ET[idx] + matrix[(i+1)*n + AT[idx].id];
        AT[idx].id = AT[idx].id ;
    }

}
__global__ void dkernel3(int *total_time , vehicle* AT,int n)
{
     unsigned idx = blockIdx.x * blockDim.x + threadIdx.x ;
    if(idx < n)
    {
        total_time[AT[idx].id]=(int)(AT[idx].time);
    }
}
//Complete the following function
void operations ( int n, int k, int m, int x, int dis, int *speed, int **results )  {
	int size = n * (k+1)	;
  int nblocks = ceil(float(float(size)/float(BLOCKSIZE))) ;  
	float *matrix , *travel_time ;
	hipMalloc(&matrix , size*sizeof(float));
	travel_time=(float*)malloc(size*sizeof(float));
  	int *gspeed ;
  	hipMalloc(&gspeed , size*sizeof(int)) ;
  	hipMemcpy(gspeed , speed, size*sizeof(int) , hipMemcpyHostToDevice);
	dkernel<<<nblocks,1024>>>(n,k,size,matrix,dis,gspeed) ;

	hipMemcpy(travel_time,matrix,size*(sizeof(float)), hipMemcpyDeviceToHost);

	struct vehicle arrival_time[n] ;
  
  int no_blocks = ceil(float(float(n)/float(BLOCKSIZE))) ;
  
  vehicle *AT;
  hipMalloc(&AT,n*sizeof(vehicle));
  dkernel1<<<no_blocks,1024>>>(matrix,AT,0,n);
  hipMemcpy(arrival_time,AT,n*sizeof(vehicle),hipMemcpyDeviceToHost);
  
  thrust::sort(arrival_time, arrival_time+n, cmp());

  float end_time[n];
  float* ET;
  hipMalloc(&ET, n*sizeof(float)) ;
  
  for(int i=0;i<k;i++)
  {
      results[0][i]=arrival_time[0].id+1;
      results[1][i]=arrival_time[n-1].id+1;
      for(int j=0;j<n;j++)
      {
          if(j<m)
          {
              end_time[j]=arrival_time[j].time + float(x) ;
          }
          else
          {
              if(end_time[j-m] > arrival_time[j].time)
              {
                  float wait_time = end_time[j-m] - arrival_time[j].time ;
                  end_time[j] = arrival_time[j].time + wait_time + float(x) ;
              }
              else
              {
                  end_time[j] = arrival_time[j].time + float(x) ;
              }
          }
      }
   
      hipMemcpy(ET , end_time, n*sizeof(float) , hipMemcpyHostToDevice);
      hipMemcpy(AT , arrival_time, n*sizeof(vehicle) , hipMemcpyHostToDevice);
      dkernel2<<<no_blocks,1024>>>(matrix , AT , ET, i , n);
      hipMemcpy(arrival_time,AT,n*sizeof(vehicle),hipMemcpyDeviceToHost);
   
      thrust::sort(arrival_time, arrival_time+n, cmp());
  }
  results[0][k]=arrival_time[0].id+1;
  results[1][k]=arrival_time[n-1].id+1;

  int *total_time ;
  hipMalloc(&total_time,n*sizeof(int));
  hipMemcpy(AT , arrival_time, n*sizeof(vehicle) , hipMemcpyHostToDevice);
  dkernel3<<<no_blocks , 1024>>>(total_time,AT,n);
  hipMemcpy(results[2],total_time,n*sizeof(int),hipMemcpyDeviceToHost);


}
int main(int argc,char **argv){

    //variable declarations
    int n,k,m,x;
    int dis;
    
    //Input file pointer declaration
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];

    inputfilepointer    = fopen( inputfilename , "r");
    
    //Checking if file ptr is NULL
    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0;
    }
    
    
    fscanf( inputfilepointer, "%d", &n );      //scaning for number of vehicles
    fscanf( inputfilepointer, "%d", &k );      //scaning for number of toll tax zones
    fscanf( inputfilepointer, "%d", &m );      //scaning for number of toll tax points
    fscanf( inputfilepointer, "%d", &x );      //scaning for toll tax zone passing time
    
    fscanf( inputfilepointer, "%d", &dis );    //scaning for distance between two consecutive toll tax zones


    // scanning for speeds of each vehicles for every subsequent toll tax combinations
    int *speed = (int *) malloc ( n*( k+1 ) * sizeof (int) );
    for ( int i=0; i<=k; i++ )  {
        for ( int j=0; j<n; j++ )  {
            fscanf( inputfilepointer, "%d", &speed[i*n+j] );
        }
    }
    
    // results is in the format of first crossing vehicles list, last crossing vehicles list 
    //               and total time taken by each vehicles to pass the highway
    int **results = (int **) malloc ( 3 * sizeof (int *) );
    results[0] = (int *) malloc ( (k+1) * sizeof (int) );
    results[1] = (int *) malloc ( (k+1) * sizeof (int) );
    results[2] = (int *) malloc ( (n) * sizeof (int) );


    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float milliseconds = 0;
    hipEventRecord(start,0);


    // Function given to implement
    operations ( n, k, m, x, dis, speed, results );


    hipDeviceSynchronize();

    hipEventRecord(stop,0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
    
    // Output file pointer declaration
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    // First crossing vehicles list
    for ( int i=0; i<=k; i++ )  {
        fprintf( outputfilepointer, "%d ", results[0][i]);
    }
    fprintf( outputfilepointer, "\n");


    //Last crossing vehicles list
    for ( int i=0; i<=k; i++ )  {
        fprintf( outputfilepointer, "%d ", results[1][i]);
    }
    fprintf( outputfilepointer, "\n");


    //Total time taken by each vehicles to pass the highway
    for ( int i=0; i<n; i++ )  {
        fprintf( outputfilepointer, "%d ", results[2][i]);
    }
    fprintf( outputfilepointer, "\n");

    fclose( outputfilepointer );
    fclose( inputfilepointer );
    return 0;
}
