cmake_minimum_required(VERSION 3.18)
project(cuda_gl_minimal LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(glfw3_DIR "${CMAKE_SOURCE_DIR}/GLFW/lib")
set(CMAKE_PREFIX_PATH ${glfw3_DIR} ${CMAKE_PREFIX_PATH})
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)

add_executable(cuda_gl_minimal
    main.cpp
    cuda_kernels.cu
    glad/glad.c
)

target_include_directories(cuda_gl_minimal PRIVATE
    glad/include
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/GLFW/include
)
target_link_directories(cuda_gl_minimal PRIVATE  "${CMAKE_SOURCE_DIR}/GLFW/lib")
target_link_libraries(cuda_gl_minimal
    OpenGL::GL
    glfw3
    ${CUDA_LIBRARIES}
)

set_target_properties(cuda_gl_minimal PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
