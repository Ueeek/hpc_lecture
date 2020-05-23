#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucketsort(int* key,int* bucket,int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>N) return;
  atomicAdd(&bucket[key[i]],1);
  __syncthreads();

  for(int accumulate=0,val=0;accumulate<=i;val++){
      key[i]=val;
      accumulate+=bucket[val];
  }
}

int main() {
  int N = 100;
  const int M=64;
  int range = 5;

  int *key;
  int *bucket;
  cudaMallocManaged(&key,N*sizeof(int));
  for (int i=0; i<N; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaMallocManaged(&bucket,range*sizeof(int));
  for(int i=0;i<range;i++)bucket[i]=0;

  bucketsort<<<(N+M-1)/M,M>>>(key,bucket,N);

  cudaDeviceSynchronize();


  for (int i=0; i<N; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
