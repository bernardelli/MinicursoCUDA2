project (Exercicio2)
cmake_minimum_required(VERSION 3.8)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/../includes)
link_directories(${CUDA_LIBRARY_DIRS}) 

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G;)  #descomentar p habilitar debug

cuda_add_executable(boxfilter boxfilter.cu)

target_link_libraries(boxfilter ${CUDA_LIBRARIES}) 