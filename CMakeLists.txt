cmake_minimum_required(VERSION 3.20)

# Force CUDA paths
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0")

project(DynamicSystemsSimulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Explicit CUDA setup
find_package(CUDAToolkit REQUIRED)

# Find vcpkg packages
find_package(glfw3 REQUIRED)
find_package(glad REQUIRED)
find_package(glm REQUIRED)
find_package(imgui REQUIRED)
find_package(OpenMP REQUIRED)

# CUDA settings
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Clear problematic flags
set(CMAKE_CUDA_FLAGS_DEBUG "")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Include CUDA headers explicitly
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include")

# Source files - UPDATED WITH 3D RENDERER
set(SOURCES
    src/main.cpp
    src/DynamicSystem3D.h
    src/gpu_evolution.cu
    src/gpu_evolution.h
    src/Renderer3D.h
    src/Renderer3D.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries including CUDA
target_link_libraries(${PROJECT_NAME} 
    glfw 
    glad::glad 
    glm::glm 
    imgui::imgui
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE 
    src/
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3>
)

# CUDA specific properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
)

# Copy any required DLLs to output directory (if needed)
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#     "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_110.dll"
#     $<TARGET_FILE_DIR:${PROJECT_NAME}>)

message(STATUS "Project configured with 3D visualization support")
message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenGL libraries: GLFW, GLAD, GLM, ImGui")
message(STATUS "Parallel computing: OpenMP + CUDA")