#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
printf("CUDA Error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
exit(EXIT_FAILURE); \
} \
} while(0)

__global__
void euclideanKernel(float*A,float*B,float*sum,int N){
int i=blockIdx.x*blockDim.x+threadIdx.x;
if(i<N){
float difference=A[i]-B[i];
atomicAdd(sum,difference*difference);
}
}

void que3(){
printf("Euclidean Distance\n");

int N=1000;
size_t size=N*sizeof(float);

float *h_A=(float*)malloc(size);
float *h_B=(float*)malloc(size);

float *d_A;
float *d_B;
float *d_sum;

srand(1000);

for(int i=0;i<N;i++){
h_A[i]=(float)rand()/RAND_MAX;
h_B[i]=(float)rand()/RAND_MAX;
}

CUDA_CHECK(cudaMalloc((void**)&d_A,size));
CUDA_CHECK(cudaMalloc((void**)&d_B,size));
CUDA_CHECK(cudaMalloc((void**)&d_sum,sizeof(float)));

CUDA_CHECK(cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice));

float zero=0.0f;

CUDA_CHECK(cudaMemcpy(d_sum,&zero,sizeof(float),cudaMemcpyHostToDevice));

int threadsPerBlock=256;
int blocksPerGrid=(N+threadsPerBlock-1)/threadsPerBlock;

printf("Vector size: %d\n",N);
printf("Threads per block: %d\n",threadsPerBlock);
printf("Number of blocks: %d\n",blocksPerGrid);

euclideanKernel<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_sum,N);

CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());

float sum;

CUDA_CHECK(cudaMemcpy(&sum,d_sum,sizeof(float),cudaMemcpyDeviceToHost));

float distance=sqrtf(sum);

printf("Euclidean Distance = %f\n",distance);

CUDA_CHECK(cudaFree(d_A));
CUDA_CHECK(cudaFree(d_B));
CUDA_CHECK(cudaFree(d_sum));

free(h_A);
free(h_B);
}

int main(){
que3();
return 0;
}

