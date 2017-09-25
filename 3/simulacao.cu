/*****
Exercício 3 do minicurso

Rafael Sturaro Bernardelli - 2017
******/

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>

#include <helper_gl.h>
#include <cuda_gl_interop.h>
#include <tiny_helper_cuda.h> 
#include <helper_math.h>

#include <gl/freeglut.h>

#define NPARTICULAS 2048	//número de partículas na simulação
#define NTHREADS 128		//número de threads por bloco
#define NTRIANGLES 20		//númetro de triângulos para desenhar cada círculo
#define RAIO_M1	0.001f		//tamanho do raio para massa = 1 u
#define TIMESTEP_MS 50		//tempo entre os callbacks 
#define WINDOWSIZE 800		//resolução da janela
#define G 0.00000005f 		//constante gravitacional fictícia

							
struct particula_s			//define a partícula
{
	float2 pos;
	float2 vel;
	float m;
} typedef Particula;



Particula* dev_part_buffer;	//ponteiro para vetor de partículas

size_t buffer_size = NPARTICULAS*NTRIANGLES*3*sizeof(float4);
							//tamanho do bufer de vértices

//usados para exibir os triângulos
//não é necessário entender como essas variáveis são empregadas!
cudaGraphicsResource_t cuda_posvbo_resource;
GLuint posVbo;

//variáveis usadas para o zoom/vizualização
float scale = 2;
float zoomTranslateX = 0;
float zoomTranslateY = 0;

float left = -1.0f;
float right = 1.0f;
float bottom = -1.0f;
float top = 1.0f;


//! define_particulas: inicializa variáveis das partículas
    /*!
	dev_part_buffer: array de partículas
	seed: usado para gerar sequência aletória
    */
__global__ void define_particulas(Particula* dev_part_buffer, int seed);


//! timer_callback: função chamada periodicamente pelo timer. Computação implementada qui
    /*!
	value: valor passado pela glutTimerFunc. Não é usado.
    */
void timer_callback(int value);

//! simula_sistema: implementa o kernel que simula gravitação
    /*!
	dev_part_buffer: array de partículas
	dev_triangle_buffer: array de vértices dos triângulos que representam as partículas

    */
__global__ void simula_sistema(Particula* dev_part_buffer, float4* dev_triangle_buffer);

//! generate_triangles: gera triângulos que se aproximam de um círculo
    /*!
	pos: centro do círculo
	m: massa da partícula (para calcular o raio)
	triangle_buffer: array de vértices + OFFSET
    */
__device__ void generate_triangles(float2 pos, float m, float4* triangle_buffer);


//! mouse_callback: função chamada toda vez que alguma variável do mouse muda. Implementa o zoom.
    /*!
	button:  botão que causou o evento
	state: estado do botão
	x, y: posição do mouse 

    */
void mouse_callback(int button, int state, int x, int y);

//! display_callback: desenha a tela
void display_callback();





int main(int argc, char ** argv)
{
	/*inicializa buffer de partículas*/

	//aloca vetor de partículas
	checkCudaErrors(cudaMalloc((void**)&dev_part_buffer, NPARTICULAS*sizeof(Particula)));
	
	//adefine os valores iniciais
	size_t n_blocks = (NPARTICULAS + NTHREADS -1)/NTHREADS;
	define_particulas<<< n_blocks, NTHREADS>>> (dev_part_buffer, time(0));
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("generate_triangles failed");
	
	/*inicializa opencv*/
	//o entendimento desse trecho não é necessário
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH| GLUT_MULTISAMPLE);
    glEnable(GL_MULTISAMPLE);
	glutInitWindowSize(WINDOWSIZE, WINDOWSIZE);
	glutCreateWindow("Exercicio 3");
	glutReportErrors();
	if (!isGLVersionSupported(2, 0))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}
	
	/*inicializa buffer de vértices do openCV e liga com o CUDA*/
	//o entendimento desse trecho não é necessário
	glGenBuffers(1, &posVbo);
	glBindBuffer(GL_ARRAY_BUFFER, posVbo);
	glBufferData(GL_ARRAY_BUFFER, buffer_size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glutReportErrors();
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_posvbo_resource, posVbo, cudaGraphicsMapFlagsWriteDiscard));
	
	/*Funções de callback*/
    glutMouseFunc(mouse_callback); 					//O callback do mouse é chamado cada vez 
													//que move-se o mouse ou clica-se um botão.
													//Ele é usado para implementar o zoom
	

	glutDisplayFunc(display_callback);				//O callback de display limpa a janela e 
													//desenha os triângulos. Basicamente, 
													//ele atualiza a tela.

	glutTimerFunc(TIMESTEP_MS,timer_callback, 0);	//Esse callback é chamado periodicamente
													//e realiza o processamento. Essa é a parte 
													//importante do código!
	
	/*inicia loop do opengl*/
	glutMainLoop();									//Coordena os callbacks até que o programa seja
													//fechado
	
	/*deleta buffers*/
	glBindBuffer(1, posVbo);
	glDeleteBuffers(1, &posVbo);
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_posvbo_resource));

	//limpa o array de particulas
    checkCudaErrors(cudaFree(dev_part_buffer));
}

//kernel que inicializa as partículas
__global__ void define_particulas(Particula* dev_part_buffer, int seed)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;

	curandState local_state;
	curand_init(seed, index, 0, &local_state);
	
	//curand_uniform retorna valor aletório entre 0.0f e 1.0f
	
	Particula particula_temp;
	
	particula_temp.pos.x = curand_uniform(&local_state) - 0.5f;
	particula_temp.pos.y = curand_uniform(&local_state) - 0.5f;

	
	
	particula_temp.vel.x = 0.0f;
	particula_temp.vel.y = 0.0f;
	
	particula_temp.m = curand_uniform(&local_state)*100 + 10;

	
	
	if (index < NPARTICULAS)
	{
		dev_part_buffer[index] = particula_temp;
	}
}




void timer_callback(int value)
{
	float4* dev_triangle_buffer;
	//pede referência do buffer para o openCV
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_posvbo_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dev_triangle_buffer, &buffer_size, cuda_posvbo_resource));

	size_t n_blocks = (NPARTICULAS + NTHREADS -1)/NTHREADS;

	simula_sistema<<<n_blocks, NTHREADS>>> (dev_part_buffer, dev_triangle_buffer);
	checkCudaErrors(cudaDeviceSynchronize());
	getLastCudaError("simula_sistema failed");



	//devolve buffer para o opencv
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_posvbo_resource, 0));
	
	//pede para atualizar o display
	glutPostRedisplay();

	//seta novo callback e timer
	glutTimerFunc(TIMESTEP_MS, timer_callback, 0);

}


__global__ void simula_sistema(Particula* dev_part_buffer, float4* dev_triangle_buffer)
{
	uint index = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (index < NPARTICULAS)
	{
		Particula particula_thread = dev_part_buffer[index];
		
		//Calcula força
		double2  = make_double2(0.0, 0.0);
		
		//IMPLEMENTAR!! calcular força
		
		//calcula nova velocidade a partir da força atual
		particula_thread.vel.x += (forca.x/particula_thread.m)*((float)TIMESTEP_MS*1e-3);
		particula_thread.vel.y += (forca.y/particula_thread.m)*((float)TIMESTEP_MS*1e-3);
		
		//Calcula a nova posicao, baseado na velocidade 
		particula_thread.pos.x += particula_thread.vel.x*((float)TIMESTEP_MS*1e-3);
		particula_thread.pos.y += particula_thread.vel.y*((float)TIMESTEP_MS*1e-3);
		

		dev_part_buffer[index] = particula_thread; //escreve a particula

		//desenha a partícula no array de vértices
		generate_triangles(particula_thread.pos, particula_thread.m, dev_triangle_buffer + index*NTRIANGLES*3);
		
	}

}



//preenche N triangulos a partir do ponteiro triangle_buffer, portanto preenche-se 3*N elementos
__device__ void generate_triangles(float2 pos, float m, float4* triangle_buffer)
{	

	float raio = RAIO_M1*sqrt(m);
	
	float sin, cos;
	//#pragma unroll NTRIANGLES //esse diretiva transforma o for em copia-cola
	for(int i = 0; i < NTRIANGLES; i++)
	{
		
		
		sincospif((2.0f*i)/(float)NTRIANGLES, &sin, &cos); //Calculate the sine and cosine of the first input argument × pi . 
		float4 temp_vertex= make_float4(pos.x + cos*raio, pos.y + sin*raio, 0.0f, 1.0f);
		
		triangle_buffer[i*3] = make_float4(pos.x, pos.y, 0.0f, 1.0f);
		triangle_buffer[i*3 + 1] = temp_vertex;
		
		triangle_buffer[((i+1)%NTRIANGLES)*3 + 2] = temp_vertex;
	}
}


//A função display_callback é chamada a cada vez que a tela é atualizada
//O entendimento dessa função não é fundamental para o exercício!
void display_callback()
{

	//define a projeção
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(left, right, bottom, top, -100.0, 100.0);

	//limpa a tela
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	//define modo de array de vértices e seleciona o nosso array
	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, posVbo);
	
	//define que estamos usando float4, sem stride e sem offset
	glVertexPointer(4, GL_FLOAT, 0, 0);

	//define a cor dos triângulos como branca
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	
	//imprime o array usando a definição de que cada três posições é um triângulo
	glDrawArrays(GL_TRIANGLES, 0, NTRIANGLES*NPARTICULAS*3);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
	glutReportErrors();
	//glPopMatrix();
}


//A função mouse_callback define o comportamento do zoom feito com a roda do mouse
//O entendimento dessa função não é fundamental para o exercício!
void mouse_callback(int button, int state, int x, int y)
{
   if ((button == 3) || (button == 4)) // é da roda do mouse
   {

		if (state == GLUT_UP) return; // é o final de um evento. igorar.
		
		//calcula novo zoom.
		float mouse_pos_x = (right - left)*(float)x / (float)WINDOWSIZE + left;
		float mouse_pos_y = (top - bottom)*(float)(WINDOWSIZE - y)/ (float)WINDOWSIZE + bottom;

		float mouse_float_x = (float)x / (float)WINDOWSIZE;
		float mouse_float_y = (float)(WINDOWSIZE-y)/ (float)WINDOWSIZE;

		int direction = (button == 3)? -1:1;

		scale += direction*scale*0.05f;

		if (scale > 15.0f)
		{
			scale = 15.0f;

		}
		else if (scale < 0.2f)
		{
			scale = 0.2f;
		}
		else 
		{


			//zoom in
			if (direction == -1)
			{
				left = mouse_pos_x - mouse_float_x*scale;
				right = mouse_pos_x + (1.0f - mouse_float_x)*scale;

				bottom = mouse_pos_y - mouse_float_y*scale;
				top = mouse_pos_y + (1.0f - mouse_float_y)*scale;
			}
			else if (direction == 1)
			{

				
				float center_x = (left + (right - left) / 2.0);
				float center_y = (bottom + (top - bottom) / 2.0);

				float ratio = scale / 15.0;
				ratio = ratio*ratio*ratio;
				
				if (center_x > 0)
				{
					right = (1.0f-ratio)*(right) + ratio*( 0.5*scale); //anda pro centro de leve

					left = right - scale;
				}
				else
				{
					left = (1.0f - ratio)*(left)+ratio*(-0.5*scale);
					right = left + scale;
				}

				if (center_y > 0)
				{
					top = (1.0f - ratio)*(top)+ratio*(0.5*scale);
					bottom = top - scale;
				}
				else
				{
					bottom = (1.0f - ratio)*(bottom)+ratio*(-0.5*scale);

					top = bottom + scale;
				}
			}
		}
	}
}