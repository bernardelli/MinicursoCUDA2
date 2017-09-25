INC=-I../includes/
GCC := g++
NVCC := nvcc -ccbin $(GCC)

GENCODE_FLAGS += -gencode arch=compute_11,code=compute_11

all: 	vectoradd

vectoradd: 	vectoradd.cu
			$(NVCC) $<  $(INC) $(GENCODE_FLAGS) -o $@ 

#nvcc -ccbin g++ vectoradd.cu -I../includes/ -o vectoradd

clean:		
			rm -rf *.o vectoradd
