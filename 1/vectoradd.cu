/*****
Exercício 1 do minicurso

Rafael Sturaro Bernardelli - 2017

-- 90% copiado dos CUDA Samples
******/


#include <cuda.h>
#include <iostream>
#include <tiny_helper_cuda.h>

using namespace std;


//! vectorAdd: implementa o kernel da soma
    /*!
		A: vetor de entrada 1
		B: vetor de entrada 2
		C: resultado da soma
		numElements: 
    */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    //IMPLEMENTAR!! calcular o índice do vetor

    if (i < numElements)
    {
       //IMPLEMENTAR!! realizar a soma
	}
}


int main(void)
{


    
    int numElements = 20000;
    size_t size = numElements * sizeof(float);


    // alloca os vetores no host
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // verifica sucesso
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        cerr << "Erro ao alocar os vetores no host" << endl;
        exit(EXIT_FAILURE);
    }

    // inicializa com valores aleatórios
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // aloca os vetores na placa
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **)&d_A, size));
    checkCudaErrors(cudaMalloc((void **)&d_B, size));
    checkCudaErrors(cudaMalloc((void **)&d_C, size));


	// copia o conteúdo do host para a placa
	checkCudaErrors(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));



    // Roda o kernel

	
	//IMPLEMENTAR!! determinar dimGrid e dimBlock
    vectorAdd<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, numElements);
    getLastCudaError("vectorAdd kernel");



	// copia o resultado para o host
    checkCudaErrors(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));


    // verifica se a soma está correta
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
           cerr << "Erro! A soma não verifica!" << endl,
            exit(EXIT_FAILURE);
        }
    }

    cout << "A soma está correta! Passa no teste." << endl;

    // libera a memória da placa
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    // libera a memória do host
    free(h_A);
    free(h_B);
    free(h_C);

    cout << "Fim" << endl;
    return 0;
}

