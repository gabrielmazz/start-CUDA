#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

__global__ void hello(){
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main(void){ 

    // Define os blocos com as threads
    hello<<<3, 8>>>();  // 2 blocos, 4 threads por bloco

    cudaDeviceSynchronize();  // Aguarda a conclusão do kernel

    return 0;
}