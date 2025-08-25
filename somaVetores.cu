#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

// Kernel na GPU: soma elemento a elemento
__global__ void somaVetoresKernel(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    // ---- Dados no host (CPU) ----
    int n = 64;                         // Tamanho do vetor
    std::vector<int> a(n), b(n), c(n);  // Vetores de entrada e saída

    // Inicialização dos vetores
    for (int i = 1; i < n; ++i) {
        a[i] = i;
        b[i] = 1000 + i;
    }

    // ---- Alocar no device (GPU) ----
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t bytes = n * sizeof(int);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // ---- Copiar CPU -> GPU ----
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // ---- Configurar e lançar kernel ----
    int threads = 256;                       // threads por bloco
    int blocks  = (n + threads - 1) / threads; // nº de blocos
    somaVetoresKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // ---- Copiar GPU -> CPU ----
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // ---- Conferir resultado ----
    for (int i = 0; i < n; ++i) {
        printf("c[%d] = %d (%d + %d)\n", i, c[i], a[i], b[i]);
    }

    // ---- Liberar ----
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
