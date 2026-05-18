#ifndef DYNAMIC_SYSTEM_3D_H
#define DYNAMIC_SYSTEM_3D_H

#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <omp.h>
#include "gpu_evolution.h"  // ← ADĂUGAT pentru GPU support

class DynamicSystem3D {
private:
    int width, height, depth;
    std::vector<int> currentState;
    std::vector<int> nextState;

    // Reguli Game of Life 3D
    int birthMin = 5;
    int birthMax = 7;
    int surviveMin = 4;
    int surviveMax = 6;

    // GPU management ← NOU
    bool gpuInitialized = false;

public:
    DynamicSystem3D(int w, int h, int d) : width(w), height(h), depth(d) {
        int totalCells = width * height * depth;
        currentState.resize(totalCells, 0);
        nextState.resize(totalCells, 0);
    }

    // Destructor pentru cleanup GPU ← NOU
    ~DynamicSystem3D() {
        cleanupGPUMemory();
    }

    inline int getIndex(int x, int y, int z) const {
        return x + y * width + z * width * height;
    }

    inline void getCoordinates(int index, int& x, int& y, int& z) const {
        x = index % width;
        y = (index / width) % height;
        z = index / (width * height);
    }

    inline int getCell(int x, int y, int z) const {
        if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth) {
            return 0;
        }
        return currentState[getIndex(x, y, z)];
    }

    inline void setCell(int x, int y, int z, int state) {
        if (x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth) {
            currentState[getIndex(x, y, z)] = state;
        }
    }

    int countNeighbors(int x, int y, int z) const {
        int count = 0;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    count += getCell(x + dx, y + dy, z + dz);
                }
            }
        }
        return count;
    }

    int applyRules(int currentState, int neighborCount) const {
        if (currentState == 1) {
            return (neighborCount >= surviveMin && neighborCount <= surviveMax) ? 1 : 0;
        }
        else {
            return (neighborCount >= birthMin && neighborCount <= birthMax) ? 1 : 0;
        }
    }

    // CPU Serial Evolution (original)
    void evolve() {
        for (int z = 0; z < depth; z++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int index = getIndex(x, y, z);
                    int neighbors = countNeighbors(x, y, z);
                    nextState[index] = applyRules(currentState[index], neighbors);
                }
            }
        }
        currentState.swap(nextState);
    }

    // CPU Parallel Evolution cu OpenMP
    void evolveParallel() {
        int totalCells = width * height * depth;

#pragma omp parallel for schedule(dynamic, 1000)
        for (int i = 0; i < totalCells; i++) {
            int x, y, z;
            getCoordinates(i, x, y, z);

            int neighbors = countNeighbors(x, y, z);
            nextState[i] = applyRules(currentState[i], neighbors);
        }

        currentState.swap(nextState);
    }

    // GPU Evolution Methods ← NOU
    void initializeGPUMemory() {
        if (!gpuInitialized) {
            initializeGPU(width, height, depth);
            gpuInitialized = true;
        }
    }

    void cleanupGPUMemory() {
        if (gpuInitialized) {
            cleanupGPU();
            gpuInitialized = false;
        }
    }

    double evolveGPU() {
        if (!gpuInitialized) {
            initializeGPUMemory();
        }

        std::vector<int> result;
        double gpuTime = evolveSystemGPU(currentState, result);
        currentState = std::move(result);
        return gpuTime;
    }

    void randomInitialize(float probability = 0.3f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int i = 0; i < currentState.size(); i++) {
            currentState[i] = (dis(gen) < probability) ? 1 : 0;
        }
    }

    int getAliveCells() const {
        int count = 0;
        for (int state : currentState) {
            count += state;
        }
        return count;
    }

    void getDimensions(int& w, int& h, int& d) const {
        w = width;
        h = height;
        d = depth;
    }

    const std::vector<int>& getCurrentState() const {
        return currentState;
    }

    void reset() {
        std::fill(currentState.begin(), currentState.end(), 0);
        std::fill(nextState.begin(), nextState.end(), 0);
    }

    // Benchmark CPU Serial
    double benchmarkEvolution(int steps) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < steps; i++) {
            evolve();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double timePerStep = duration.count() / (double)steps / 1000.0;
        return timePerStep;
    }

    // Benchmark CPU Parallel
    double benchmarkEvolutionParallel(int steps) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < steps; i++) {
            evolveParallel();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double timePerStep = duration.count() / (double)steps / 1000.0;
        return timePerStep;
    }

    // Benchmark GPU ← NOU
    double benchmarkEvolutionGPU(int steps) {
        if (!gpuInitialized) {
            initializeGPUMemory();
        }

        double totalTime = 0.0;
        for (int i = 0; i < steps; i++) {
            totalTime += evolveGPU();
        }

        return totalTime / steps;
    }

    // CPU Comparison (original)
    void compareCPUPerformance(int steps = 10) {
        std::vector<int> savedState = currentState;

        std::cout << "\n=== CPU Performance Comparison ===" << std::endl;
        std::cout << "Grid size: " << width << "x" << height << "x" << depth << std::endl;
        std::cout << "Steps: " << steps << std::endl;

        currentState = savedState;
        double serialTime = benchmarkEvolution(steps);

        currentState = savedState;
        double parallelTime = benchmarkEvolutionParallel(steps);

        double speedup = serialTime / parallelTime;
        int numThreads = omp_get_max_threads();

        std::cout << "CPU Serial:   " << serialTime << " ms/step" << std::endl;
        std::cout << "CPU Parallel: " << parallelTime << " ms/step (" << numThreads << " threads)" << std::endl;
        std::cout << "Speedup:      " << speedup << "x" << std::endl;
        std::cout << "Efficiency:   " << (speedup / numThreads * 100) << "%" << std::endl;

        currentState = savedState;
    }

    // Comprehensive Performance Comparison - ALL METHODS ← NOU
    void compareAllMethods(int steps = 10) {
        std::vector<int> savedState = currentState;

        std::cout << "\n=== COMPREHENSIVE PERFORMANCE COMPARISON ===" << std::endl;
        std::cout << "Grid size: " << width << "x" << height << "x" << depth
            << " (" << (width * height * depth) << " cells)" << std::endl;
        std::cout << "Steps: " << steps << std::endl;

        // CPU Serial
        std::cout << "\nTesting CPU Serial..." << std::endl;
        currentState = savedState;
        double serialTime = benchmarkEvolution(steps);

        // CPU Parallel
        std::cout << "Testing CPU Parallel..." << std::endl;
        currentState = savedState;
        double parallelTime = benchmarkEvolutionParallel(steps);

        // GPU CUDA
        std::cout << "Testing GPU CUDA..." << std::endl;
        currentState = savedState;
        double gpuTime = benchmarkEvolutionGPU(steps);

        // Calculate speedups
        double cpuSpeedup = serialTime / parallelTime;
        double gpuSpeedup = serialTime / gpuTime;
        double gpuVsCpuParallel = parallelTime / gpuTime;

        // Results
        std::cout << "\n=== RESULTS ===" << std::endl;
        std::cout << "CPU Serial:      " << serialTime << " ms/step" << std::endl;
        std::cout << "CPU Parallel:    " << parallelTime << " ms/step (" << omp_get_max_threads() << " threads)" << std::endl;
        std::cout << "GPU CUDA:        " << gpuTime << " ms/step" << std::endl;

        std::cout << "\n=== SPEEDUPS ===" << std::endl;
        std::cout << "CPU Parallel vs Serial:  " << cpuSpeedup << "x" << std::endl;
        std::cout << "GPU vs Serial:           " << gpuSpeedup << "x" << std::endl;
        std::cout << "GPU vs CPU Parallel:     " << gpuVsCpuParallel << "x" << std::endl;

        // Throughput analysis
        double cellsPerSecond = (width * height * depth) / (gpuTime / 1000.0);
        std::cout << "\n=== THROUGHPUT ===" << std::endl;
        std::cout << "GPU Throughput: " << cellsPerSecond / 1000000.0 << " million cells/second" << std::endl;

        // Efficiency analysis
        std::cout << "\n=== EFFICIENCY ===" << std::endl;
        std::cout << "CPU Parallel Efficiency: " << (cpuSpeedup / omp_get_max_threads() * 100) << "%" << std::endl;

        // Performance per watt estimate
        std::cout << "\n=== PERFORMANCE ANALYSIS ===" << std::endl;
        if (gpuSpeedup > 50) {
            std::cout << "GPU shows EXCELLENT speedup (>50x)" << std::endl;
        }
        else if (gpuSpeedup > 10) {
            std::cout << "GPU shows GOOD speedup (>10x)" << std::endl;
        }
        else {
            std::cout << "GPU speedup is modest - consider larger grids" << std::endl;
        }

        currentState = savedState;
    }
};

#endif // DYNAMIC_SYSTEM_3D_H