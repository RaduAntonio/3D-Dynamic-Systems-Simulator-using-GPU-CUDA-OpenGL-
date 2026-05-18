#ifndef GPU_EVOLUTION_H
#define GPU_EVOLUTION_H

#include <vector>

// Initialize GPU memory and resources
void initializeGPU(int width, int height, int depth);

// Cleanup GPU resources
void cleanupGPU();

// Evolve system on GPU (includes memory transfer time)
double evolveSystemGPU(const std::vector<int>& currentState, std::vector<int>& nextState);

// Evolve system on GPU (optimized, excludes transfer time)
double evolveSystemGPUOptimized(const std::vector<int>& currentState, std::vector<int>& nextState);

#endif // GPU_EVOLUTION_H