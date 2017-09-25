INC=-I../includes/
GCC := g++


CUDA_PATH       ?= /usr/local/cuda-6.5

NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(GCC)

GENCODE_FLAGS += -gencode arch=compute_11,code=compute_11


all: 	boxfilter

boxfilter: 	boxfilter.cu
			$(NVCC) $<  $(INC) $(GENCODE_FLAGS) -o $@ 

clean:		
			rm -rf *.o boxfilter


