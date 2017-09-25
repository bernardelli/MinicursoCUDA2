project (Exercicio3)
cmake_minimum_required(VERSION 3.8)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/../includes)
link_directories(${CUDA_LIBRARY_DIRS}) 

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G;)  

cuda_add_executable(simulacao simulacao.cu)

IF (WIN32)
add_library(glew SHARED IMPORTED)
set_property(TARGET glew PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/bin/glew64.dll)
set_property(TARGET glew PROPERTY IMPORTED_IMPLIB ${PROJECT_SOURCE_DIR}/lib/glew64.lib)

add_library(glut SHARED IMPORTED)
set_property(TARGET glut PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/bin/freeglut.dll)
set_property(TARGET glut PROPERTY IMPORTED_IMPLIB ${PROJECT_SOURCE_DIR}/lib/freeglut.lib)


target_link_libraries(simulacao glut glew ${CUDA_LIBRARIES}) 
ELSE()
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
target_link_libraries(simulacao ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES}) 
ENDIF()
	
