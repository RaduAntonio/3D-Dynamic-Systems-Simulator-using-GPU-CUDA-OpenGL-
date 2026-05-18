# Dynamic Systems Simulator using GPU

A high-performance 3D cellular automaton simulator implementing **Conway's Game of Life** in three dimensions, with real-time OpenGL visualization and parallel execution on both CPU (OpenMP) and GPU (CUDA).

> Bachelor's Thesis — Faculty of Electronics, Telecommunications and Information Technology  
> National University of Science and Technology POLITEHNICA Bucharest  
> Author: **Antonio-Nicolas Radu** | Advisor: Ș.L.Dr.Ing. George Valentin Stoica | 2025

---

## Preview

| 3D Simulation (low density) | 3D Simulation (high density) |
|---|---|
| Blue voxels — low population | Green voxels — medium/high population |

The voxel color changes dynamically based on population density: **blue** (sparse) → **green** (medium) → **red** (dense).

---

## Features

- **3D Game of Life** with Moore neighborhood (26 neighbors per cell), rules Birth 5–7 / Survive 4–6
- **Three execution backends**:
  - CPU Serial — sequential reference implementation
  - CPU Parallel — OpenMP with dynamic scheduling
  - GPU CUDA — massively parallel kernel on NVIDIA RTX
- **Interactive 3D visualization** via OpenGL 3.3 Core Profile with Phong lighting
- **FPS-style camera** (WASD + mouse look, toggled with TAB)
- **Instanced rendering** — all live voxels drawn in a single draw call
- **Real-time benchmarking panel** (ImGui) with live speedup and throughput stats
- **Auto performance rating** — GOOD / EXCELLENT based on measured GPU speedup
- **Configurable grid size** from 32³ to 128³ via slider (with automatic GPU memory reallocation)

---

## Performance Results

Tested on **Intel i5-10500H (12 threads)** + **NVIDIA RTX 3060 Laptop GPU** (Ampere, CC 8.6, 30 SMs):

| Grid Size | Cells | CPU Serial | CPU Parallel | GPU CUDA | GPU Speedup vs Serial |
|-----------|-------|-----------|--------------|----------|-----------------------|
| 32³ | 32,768 | 8.4 ms | 1.5 ms | 0.19 ms | ~44x |
| 64³ | 262,144 | 69.5 ms | 13.0 ms | 0.75 ms | ~95x |
| 96³ | 884,736 | 246.8 ms | 44.9 ms | 4.65 ms | ~85x |
| 128³ | 2,097,152 | 582.7 ms | 184.3 ms | 5.35 ms | ~109x |

**GPU Throughput** peaks at **~13,315 million cells/second** for 128³ grids.  
**CPU Parallel efficiency**: ~47–53% (12 threads, dynamic scheduling).

---

## Architecture

```
DynamicSystemsSimulator/
├── src/
│   ├── DynamicSystem3D.h       # Core simulation engine (serial, OpenMP, GPU interface)
│   ├── gpu_evolution.h         # CUDA interface declarations
│   ├── gpu_evolution.cu        # CUDA kernels + GPUEvolutionManager class
│   ├── Renderer3D.h/.cpp       # OpenGL renderer, instanced voxels, FPS camera
│   └── main.cpp                # App loop, ImGui UI, performance tracking
└── CMakeLists.txt              # Build config (CUDA 13.0, OpenGL, vcpkg)
```

**Key design patterns:**
- **Double buffering** — `currentState` / `nextState` for race-condition-free parallel updates
- **RAII** — `GPUEvolutionManager` destructor ensures automatic GPU memory cleanup
- **Separation of concerns** — compute layer, graphics layer, and application layer are fully decoupled
- **Linear 1D indexing** for cache-friendly 3D grid access: `idx = x + y*width + z*width*height`

---

## Build Requirements

| Dependency | Version |
|------------|---------|
| NVIDIA GPU | CUDA Compute Capability ≥ 7.0 |
| CUDA Toolkit | 12.x / 13.0 |
| Visual Studio | 2022 (MSVC) |
| CMake | ≥ 3.18 |
| vcpkg | latest |
| OpenGL | 3.3 Core Profile |
| Libraries | GLFW, GLAD, GLM, ImGui (via vcpkg) |

---

## Build & Run

```bash
# Clone the repository
git clone https://github.com/<your-username>/DynamicSystemsSimulator.git
cd DynamicSystemsSimulator

# Install dependencies via vcpkg (adjust path as needed)
vcpkg install glfw3 glad glm imgui

# Configure and build
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release

# Run
./build/Release/DynamicSystemsSimulator.exe
```

---

## Controls

| Input | Action |
|-------|--------|
| `TAB` | Toggle mouse look (FPS camera) |
| `W / A / S / D` | Move camera |
| `Q / E` | Move camera up / down |
| Mouse | Look around (when mouse look is active) |
| Scroll wheel | Zoom |
| ImGui panel | Select evolution method, run benchmarks, adjust grid size |

---

## CUDA Implementation Details

- **Block size**: 256 threads — optimized for Ampere SM occupancy (8 × 32 warps)
- **Grid size**: `(totalCells + 255) / 256` blocks
- **Memory layout**: flat 1D `int*` arrays for coalesced global memory access
- **Device function** `countNeighbors_device` — inline neighbor counting on-chip
- **No shared memory tiling** in the base version (listed as a future optimization)
- GPU memory allocated once at initialization, reallocated only on grid size change

---

## Scalability Analysis

- **CPU Serial**: O(N) per generation, strictly linear with cell count
- **CPU Parallel**: ~5–5.7x speedup on 12 threads; efficiency capped by memory bandwidth
- **GPU**: sub-linear scaling — throughput *increases* with grid size as SM utilization grows from ~20% (32³) to ~85% (128³)
- Crossover point (GPU > CPU Parallel): already at 32³ and below

---

## Roadmap / Future Work

- [ ] Shared memory tiling in CUDA kernel (estimated 2–5x additional speedup)
- [ ] Multiple CUDA streams for overlap of compute and memory transfers
- [ ] Level-of-Detail (LOD) rendering for larger grids
- [ ] Multi-GPU support via CUDA + MPI
- [ ] Interactive pattern editor and 3D pattern library
- [ ] Statistical analysis overlay (entropy, cluster detection)
- [ ] Support for alternative cellular automaton rulesets

---

## References

- Cheng, Grossman, McKercher — *Professional CUDA C Programming*, Wiley, 2014
- Kirk & Hwu — *Programming Massively Parallel Processors*, 3rd ed., Morgan Kaufmann, 2016
- NVIDIA — [CUDA C++ Programming Guide v12](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- Khronos Group — [OpenGL 4.6 Core Profile Specification](https://www.khronos.org/registry/OpenGL/)
- Conway, J.H. — *The Game of Life*, Scientific American, 1970

---

## License

This project was developed as a bachelor's thesis at UNSTPB. Source code is provided for academic reference.
