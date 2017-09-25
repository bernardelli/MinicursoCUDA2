/*****
Exercício 2 do minicurso

Rafael Sturaro Bernardelli - 2017
******/



#include <cuda.h>
#include <tiny_helper_cuda.h>
#include <tiny_helper_image.h>
#include <iostream>

#define nullptr NULL

using namespace std;

//! boxfilter_x: implementa o kernel do boxfilter na direção x
    /*!
	in : imagem de entrada
	out : imagem de saída
	N : tamanho do filtro
	width : largura da imagem

    */
__global__ void boxfilter_x(unsigned char * in, unsigned char * out, int N, int width);

//! clamp_zero_to_L: limita n entre 0 e L
    /*!
	n : valor a ser grampeado
	L : limite máximo para n
    */
__device__ __inline__ int clamp_zero_to_L(int n, int L);

int main(int argc, char ** argv)
{

    //entrada dos argumentos
    if(argc != 4)
    {
        cout << "Uso: boxfilter [tamanho do boxfilter] arquivo.pgm arquivo_out.pgm" << endl;
        return 1;
    }
    
	
	//pega o tamanho do filtro
    int N = atoi(argv[1]); 
    
	
	//verifica se N é válido
    if (N<3 || N>100 || N%2 == 0)
    {
        cout << "tamanho do boxfilter deve ser ímpar, " << endl;
        return 1;
    }
    
    
    char * input_file = argv[2];
    unsigned char * host_image_buffer = nullptr;
    unsigned int  width, height, channels;
    
	//faz a leitura da imagem num vetor de unsigned char*. A função já aloca memória necessária 
	cout << "Começa a ler o arquivo..." << endl;
    bool status = __loadPPM(input_file, &host_image_buffer, &width, &height, &channels);
    
	
	//verifica o sucesso da leitura
    if(status == true)
    {
        cout << "Arquivo "<< input_file << " lido corretamente." << endl;
    }
    else
    {
        cerr << "Erro ao ler arquivo "<< input_file << endl;
        return 1;
    }

	//verifica se a imagem tem só um canal (imagem escala de cinza)
    if(channels != 1)
    {
        cerr << "Erro! imagem deve ser preto e braco!" << endl;
        return 1;
    }
    
	//tamanho em bytes da imagem lida
    size_t size = width*height*sizeof(unsigned char);
    
    //Aloca vetor de imagem de entrada e saída no CUDA
    unsigned char * dev_image_in;
    unsigned char * dev_image_out;
	//IMPLEMENTAR!! alocar vetores
    
    //copia imagem de entrada para a GPU

	cout << "Começa cópia para a GPU..." << endl;
    checkCudaErrors(cudaMemcpy(dev_image_in, host_image_buffer, size, cudaMemcpyHostToDevice));

	cout << "Cópia finalizada" << endl;

    
    //lança o kernel do filtro
    dim3 dimBlock(128, 1); 
	
	size_t shared_buf_size = (dimBlock.x + N -1)*sizeof(unsigned char);
	
    dim3 dimGrid;
	dimGrid.x = (width + dimBlock.x -1 )/dimBlock.x;
	dimGrid.y = height;
    
	cout << "Lança kernel..." << endl;

	boxfilter_x <<<dimGrid, dimBlock, shared_buf_size >>> (dev_image_in, dev_image_out, N, width);
    getLastCudaError("boxfilter_x kernel");

	cout << "Kernel concluído." << endl;


    //Copia resultado da filtragem para o host
    checkCudaErrors(cudaMemcpy(host_image_buffer, dev_image_out, size, cudaMemcpyDeviceToHost));

    //Escreve resultado da filtragem num arquivo
    char * output_file = argv[3]; //pega o nome do arquivo de saída
    
	cout << "Começa a escrever o arquivo..." << endl;
	status = __savePPM(output_file, host_image_buffer, width, height, channels);

	if (status == true)
	{
		cout << "Arquivo " << output_file << " salvo corretamente." << endl;
	}
	else
	{
		cerr << "Erro ao salvar arquivo " << output_file << endl;
	}
    
    //libera array na RAM
    free(host_image_buffer);
	
    //libera arrays alocados na GPU
    checkCudaErrors(cudaFree(dev_image_in));
    checkCudaErrors(cudaFree(dev_image_out));
   
	return 0;
}


__global__ void boxfilter_x(unsigned char * in, unsigned char * out, int N, int width)
{

	//IMPLEMENTAR!! calcular posição x e y da imagem
	int x = ;
	int y = ;
	
	//declara o buffer compartilhado
    extern  __shared__ unsigned char shared_buffer [];
	

	//preenche o centro buffer
	int idx = clamp_zero_to_L(x, width) + y*width;
	shared_buffer[?????] = in[idx]; //IMPLEMENTAR!! calcular índice do shared_buffer
	
	
	if(threadIdx.x < N/2) //preenche a borda esquerda do buffer
	{
		int idx2 = clamp_zero_to_L(x - N/2, width) + y*width;
		shared_buffer[?????]  = in[idx2]; //IMPLEMENTAR!! calcular índice do shared_buffer
	}
	else if ((blockDim.x-threadIdx.x) <= N/2) //preenche a borda direita do buffer
	{
		int idx2 = clamp_zero_to_L(x + N/2, width) + y*width;
		shared_buffer[?????]  = in[idx2]; //IMPLEMENTAR!! calcular índice do shared_buffer
	}

	//sincroniza bloco depois de preencher memória compartilhada
	__syncthreads();
	
	int acc = 0;
	
	//IMPLEMENTAR!! calcular a soma dos N elementos através de um for
		
	//escreve resultado
	if(x < width)
	{
		out[idx] = acc/N;
	}
	
}

__device__ __inline__ int clamp_zero_to_L(int n, int L)
{
	L--;
	n = n>L ? L : n;
	return n<0 ? 0 : n;
}

