#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
do { \
cudaError_t err=call; \
if(err!=cudaSuccess){ \
printf("CUDA Error at %s:%d -> %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); \
exit(EXIT_FAILURE); \
} \
} while(0)

__global__
void saxpyKernel(float *x,float *y,float alpha,int N){
int i=blockIdx.x*blockDim.x+threadIdx.x;
if(i<N){
y[i]=alpha*x[i]+y[i];
}
}

void additionalExercise1(){
printf("\nLINEAR ALGEBRA: y = alpha*x + y\n");

int N=256;
float alpha=2.0f;

size_t size=N*sizeof(float);

float *h_x=(float*)malloc(size);
float *h_y=(float*)malloc(size);

float *d_x;
float *d_y;

for(int i=0;i<N;i++){
h_x[i]=(float)i;
h_y[i]=10.0f;
}

printf("Alpha = %.2f\n",alpha);

printf("\nBefore operation:\n");

for(int i=0;i<10;i++){
printf("x[%d] = %.2f, y[%d] = %.2f\n",i,h_x[i],i,h_y[i]);
}

CUDA_CHECK(cudaMalloc((void**)&d_x,size));
CUDA_CHECK(cudaMalloc((void**)&d_y,size));

CUDA_CHECK(cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(d_y,h_y,size,cudaMemcpyHostToDevice));

int threadsPerBlock=256;
int blocksPerGrid=(N+threadsPerBlock-1)/threadsPerBlock;

printf("\nThreads per block : %d\n",threadsPerBlock);
printf("Number of blocks  : %d\n",blocksPerGrid);

saxpyKernel<<<blocksPerGrid,threadsPerBlock>>>(d_x,d_y,alpha,N);

CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());

CUDA_CHECK(cudaMemcpy(h_y,d_y,size,cudaMemcpyDeviceToHost));

printf("\nAfter operation y = alpha*x + y:\n");

for(int i=0;i<10;i++){
printf("y[%d] = %.2f\n",i,h_y[i]);
}

CUDA_CHECK(cudaFree(d_x));
CUDA_CHECK(cudaFree(d_y));

free(h_x);
free(h_y);
}

int main(){
additionalExercise1();
return 0;
}

