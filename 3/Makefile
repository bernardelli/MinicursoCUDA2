GLPATH=/usr/lib
INC=-I../includes/
GCC := g++

# Location of the CUDA Toolkit
CUDA_PATH       ?= /usr/local/cuda-6.5


NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(GCC)




SAMPLE_ENABLED := 1

# Makefile include to help find GL Libraries
include ./findgllib.mk


LIBRARIES := -L./lib/ #$(GLLINK)
LIBRARIES += -lGL -lGLU -lX11 -lXi -lXmu -lglut -lGLEW

GENCODE_FLAGS := -gencode arch=compute_11,code=compute_11



all: 	simulacao

simulacao: 	simulacao.cu
			$(EXEC) $(NVCC) $(GENCODE_FLAGS) $(INC) -o $@ $< $(LIBRARIES)

clean:
			rm -rf *.o simulacao
