project (Exercicio1)
cmake_minimum_required(VERSION 3.8)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/../includes)
link_directories(${CUDA_LIBRARY_DIRS}) 

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G;)  

cuda_add_executable(vectoradd vectoradd.cu)

target_link_libraries(vectoradd ${CUDA_LIBRARIES}) 