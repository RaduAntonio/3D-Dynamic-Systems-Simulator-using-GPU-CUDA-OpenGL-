#include "gpu_evolution.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Device function pentru calculul vecinilor
__device__ int countNeighbors_device(const int* grid, int x, int y, int z, int width, int height, int depth) {
    int count = 0;

    // Moore neighborhood 3D (26 vecini)
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue; // Skip current cell

                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;

                // Boundary check
                if (nx >= 0 && nx < width && ny >= 0 && ny < height && nz >= 0 && nz < depth) {
                    int index = nx + ny * width + nz * width * height;
                    count += grid[index];
                }
            }
        }
    }

    return count;
}

// Device function pentru aplicarea regulilor Game of Life 3D
__device__ int applyRules_device(int currentState, int neighborCount) {
    // Birth: 5-7 neighbors, Survive: 4-6 neighbors
    if (currentState == 1) {
        // Cell is alive - check survival
        return (neighborCount >= 4 && neighborCount <= 6) ? 1 : 0;
    }
    else {
        // Cell is dead - check birth
        return (neighborCount >= 5 && neighborCount <= 7) ? 1 : 0;
    }
}

// CUDA Kernel pentru evoluția sistemului
__global__ void evolution_kernel(const int* currentState, int* nextState, int width, int height, int depth) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCells = width * height * depth;

    if (idx >= totalCells) return;

    // Convert 1D index to 3D coordinates
    int x = idx % width;
    int y = (idx / width) % height;
    int z = idx / (width * height);

    // Count neighbors and apply rules
    int neighbors = countNeighbors_device(currentState, x, y, z, width, height, depth);
    nextState[idx] = applyRules_device(currentState[idx], neighbors);
}

// GPU Evolution Manager Class
class GPUEvolutionManager {
private:
    int* d_currentState;
    int* d_nextState;
    int width, height, depth;
    int totalCells;
    size_t stateSize;
    bool initialized;

public:
    GPUEvolutionManager() : d_currentState(nullptr), d_nextState(nullptr), initialized(false) {}

    ~GPUEvolutionManager() {
        cleanup();
    }

    void initialize(int w, int h, int d) {
        cleanup(); // Clean up any previous allocation

        width = w;
        height = h;
        depth = d;
        totalCells = width * height * depth;
        stateSize = totalCells * sizeof(int);

        // Allocate GPU memory
        CUDA_CHECK(cudaMalloc(&d_currentState, stateSize));
        CUDA_CHECK(cudaMalloc(&d_nextState, stateSize));

        initialized = true;

        std::cout << "GPU memory allocated: " << (stateSize * 2) / (1024 * 1024) << " MB" << std::endl;
    }

    void cleanup() {
        if (d_currentState) {
            cudaFree(d_currentState);
            d_currentState = nullptr;
        }
        if (d_nextState) {
            cudaFree(d_nextState);
            d_nextState = nullptr;
        }
        initialized = false;
    }

    double evolveGPU(const std::vector<int>& hostState, std::vector<int>& result) {
        if (!initialized) {
            std::cerr << "GPU manager not initialized!" << std::endl;
            return 0.0;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Copy data to GPU
        CUDA_CHECK(cudaMemcpy(d_currentState, hostState.data(), stateSize, cudaMemcpyHostToDevice));

        // Calculate grid and block dimensions
        int blockSize = 256; // Good for RTX 3060 Ti
        int gridSize = (totalCells + blockSize - 1) / blockSize;

        // Launch kernel
        evolution_kernel << <gridSize, blockSize >> > (d_currentState, d_nextState, width, height, depth);

        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());

        // Wait for kernel to complete
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back to host
        result.resize(totalCells);
        CUDA_CHECK(cudaMemcpy(result.data(), d_nextState, stateSize, cudaMemcpyDeviceToHost));

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        return duration.count() / 1000.0; // Return time in milliseconds
    }

    // Optimized version with persistent GPU memory
    double evolveGPUOptimized(const std::vector<int>& hostState, std::vector<int>& result) {
        if (!initialized) {
            std::cerr << "GPU manager not initialized!" << std::endl;
            return 0.0;
        }

        // Only time the computation, not memory transfers
        CUDA_CHECK(cudaMemcpy(d_currentState, hostState.data(), stateSize, cudaMemcpyHostToDevice));

        auto start = std::chrono::high_resolution_clock::now();

        // Calculate grid and block dimensions
        int blockSize = 256;
        int gridSize = (totalCells + blockSize - 1) / blockSize;

        // Launch kernel
        evolution_kernel << <gridSize, blockSize >> > (d_currentState, d_nextState, width, height, depth);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMemcpy(result.data(), d_nextState, stateSize, cudaMemcpyDeviceToHost));

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / 1000.0;
    }

    void printGPUInfo() {
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

        if (deviceCount > 0) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

            std::cout << "\n=== GPU Information ===" << std::endl;
            std::cout << "Device: " << prop.name << std::endl;
            std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
            std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
            std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        }
    }
};

// Global GPU manager instance
static GPUEvolutionManager gpuManager;

// C++ interface functions
void initializeGPU(int width, int height, int depth) {
    gpuManager.initialize(width, height, depth);
    gpuManager.printGPUInfo();
}

void cleanupGPU() {
    gpuManager.cleanup();
}

double evolveSystemGPU(const std::vector<int>& currentState, std::vector<int>& nextState) {
    return gpuManager.evolveGPU(currentState, nextState);
}

double evolveSystemGPUOptimized(const std::vector<int>& currentState, std::vector<int>& nextState) {
    return gpuManager.evolveGPUOptimized(currentState, nextState);
}