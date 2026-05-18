#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <omp.h>
#include "DynamicSystem3D.h"
#include "Renderer3D.h"

// Simple performance tracking struct
struct SimplePerformanceStats {
    int sampleCount = 0;
    double totalCPUSerialTime = 0.0;
    double totalCPUParallelTime = 0.0;
    double totalGPUTime = 0.0;
    double bestGPUSpeedup = 0.0;
    double worstGPUSpeedup = 999999.0;
    double bestCPUSpeedup = 0.0;
    double totalGPUSpeedup = 0.0;
    double totalCPUSpeedup = 0.0;

    void addSample(double cpuSerial, double cpuParallel, double gpu) {
        if (cpuSerial > 0) {
            totalCPUSerialTime += cpuSerial;

            if (gpu > 0) {
                double gpuSpeedup = cpuSerial / gpu;
                totalGPUTime += gpu;
                totalGPUSpeedup += gpuSpeedup;

                if (gpuSpeedup > bestGPUSpeedup) bestGPUSpeedup = gpuSpeedup;
                if (gpuSpeedup < worstGPUSpeedup) worstGPUSpeedup = gpuSpeedup;
            }

            if (cpuParallel > 0) {
                double cpuSpeedup = cpuSerial / cpuParallel;
                totalCPUParallelTime += cpuParallel;
                totalCPUSpeedup += cpuSpeedup;

                if (cpuSpeedup > bestCPUSpeedup) bestCPUSpeedup = cpuSpeedup;
            }

            sampleCount++;
        }
    }

    double getAverageGPUSpeedup() const {
        return sampleCount > 0 ? totalGPUSpeedup / sampleCount : 0.0;
    }

    double getAverageCPUSpeedup() const {
        return sampleCount > 0 ? totalCPUSpeedup / sampleCount : 0.0;
    }

    double getGPUThroughput(int gridSize) const {
        if (sampleCount == 0 || totalGPUTime == 0) return 0.0;
        double avgGPUTime = totalGPUTime / sampleCount;
        double totalCells = static_cast<double>(gridSize) * gridSize * gridSize;
        return (totalCells / 1000000.0) / (avgGPUTime / 1000.0);
    }

    double getCPUEfficiency() const {
        double avgSpeedup = getAverageCPUSpeedup();
        int numThreads = omp_get_max_threads();
        return (avgSpeedup / numThreads) * 100.0;
    }

    void clear() {
        sampleCount = 0;
        totalCPUSerialTime = 0.0;
        totalCPUParallelTime = 0.0;
        totalGPUTime = 0.0;
        bestGPUSpeedup = 0.0;
        worstGPUSpeedup = 999999.0;
        bestCPUSpeedup = 0.0;
        totalGPUSpeedup = 0.0;
        totalCPUSpeedup = 0.0;
    }

    void printReport(int gridSize) const {
        std::cout << "\n========== PERFORMANCE STATISTICS REPORT ==========" << std::endl;
        std::cout << "Grid Size: " << gridSize << "x" << gridSize << "x" << gridSize << std::endl;
        std::cout << "Samples collected: " << sampleCount << std::endl;
        std::cout << "CPU threads: " << omp_get_max_threads() << std::endl;

        if (sampleCount > 0) {
            std::cout << "\n--- GPU PERFORMANCE ---" << std::endl;
            std::cout << "Average GPU speedup: " << getAverageGPUSpeedup() << "x vs CPU Serial" << std::endl;
            std::cout << "Best GPU speedup: " << bestGPUSpeedup << "x" << std::endl;
            std::cout << "Worst GPU speedup: " << worstGPUSpeedup << "x" << std::endl;
            std::cout << "GPU Throughput: " << getGPUThroughput(gridSize) << " million cells/sec" << std::endl;

            std::cout << "\n--- CPU PARALLEL PERFORMANCE ---" << std::endl;
            std::cout << "Average CPU Parallel speedup: " << getAverageCPUSpeedup() << "x vs Serial" << std::endl;
            std::cout << "Best CPU Parallel speedup: " << bestCPUSpeedup << "x" << std::endl;
            std::cout << "CPU Parallel efficiency: " << getCPUEfficiency() << "%" << std::endl;

            std::cout << "\n--- PERFORMANCE RATING ---" << std::endl;
            double avgGPUSpeedup = getAverageGPUSpeedup();
            if (avgGPUSpeedup > 100) std::cout << "GPU Rating: EXCELLENT (>100x)" << std::endl;
            else if (avgGPUSpeedup > 50) std::cout << "GPU Rating: VERY GOOD (50-100x)" << std::endl;
            else if (avgGPUSpeedup > 20) std::cout << "GPU Rating: GOOD (20-50x)" << std::endl;
            else if (avgGPUSpeedup > 5) std::cout << "GPU Rating: MODEST (5-20x)" << std::endl;
            else std::cout << "GPU Rating: NEEDS IMPROVEMENT (<5x)" << std::endl;

            std::cout << "\n--- RECOMMENDATIONS ---" << std::endl;
            if (avgGPUSpeedup < 20) {
                std::cout << "• Try larger grid sizes for better GPU utilization" << std::endl;
            }
            if (getCPUEfficiency() < 70) {
                std::cout << "• CPU parallel efficiency could be improved" << std::endl;
            }
            std::cout << "• Best method for this grid size: " << (avgGPUSpeedup > 10 ? "GPU" : "CPU Parallel") << std::endl;
        }
        std::cout << "====================================================" << std::endl;
    }
};

// Global variables for mouse handling
bool firstMouse = true;
float lastX = 800.0f / 2.0f;
float lastY = 600.0f / 2.0f;
Renderer3D* globalRenderer = nullptr;
bool cameraControlEnabled = false;
bool mouseLookToggled = false;

// Keyboard input for mouse control toggle
void processKeyboardInput(GLFWwindow* window) {
    static bool tabPressed = false;

    if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS) {
        if (!tabPressed) {
            cameraControlEnabled = !cameraControlEnabled;
            mouseLookToggled = true;

            if (cameraControlEnabled) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                firstMouse = true;
                std::cout << "Mouse Look ENABLED (TAB to toggle)" << std::endl;
            }
            else {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                std::cout << "Mouse Look DISABLED (TAB to toggle)" << std::endl;
            }

            tabPressed = true;
        }
    }
    else {
        tabPressed = false;
    }
}

// Mouse callback
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (!cameraControlEnabled || !globalRenderer) return;

    if (firstMouse) {
        lastX = static_cast<float>(xpos);
        lastY = static_cast<float>(ypos);
        firstMouse = false;
    }

    float xoffset = static_cast<float>(xpos - lastX);
    float yoffset = static_cast<float>(lastY - ypos);

    lastX = static_cast<float>(xpos);
    lastY = static_cast<float>(ypos);

    globalRenderer->processMouseMovement(xoffset, yoffset);
}

// Scroll callback
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (globalRenderer) {
        globalRenderer->processMouseScroll(static_cast<float>(yoffset));
    }
}

int main() {
    std::cout << "Starting Dynamic Systems Simulator with Performance Tracking..." << std::endl;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(1600, 1000, "Dynamic Systems Simulator - 3D Visualization + Performance", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Print system info
    std::cout << "=== System Information ===" << std::endl;
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GPU: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "OpenMP threads available: " << omp_get_max_threads() << std::endl;

    // Initialize system
    int grid_size = 64;
    DynamicSystem3D system(grid_size, grid_size, grid_size);
    system.randomInitialize(0.2f);
    int alive_cells = system.getAliveCells();

    // Initialize 3D Renderer
    Renderer3D renderer;
    globalRenderer = &renderer;
    if (!renderer.initialize(grid_size, grid_size, grid_size)) {
        std::cerr << "Failed to initialize 3D renderer" << std::endl;
        return -1;
    }

    // Performance tracking variables
    int evolution_method = 0;
    float cpu_serial_time = 0.0f;
    float cpu_parallel_time = 0.0f;
    float gpu_cuda_time = 0.0f;
    float cpu_speedup = 0.0f;
    float gpu_speedup = 0.0f;
    int num_threads = omp_get_max_threads();

    // Performance statistics - NEW!
    SimplePerformanceStats perfStats;

    // 3D Visualization settings
    bool show_3d_view = true;
    bool auto_evolve = false;
    float evolution_speed = 2.0f;
    float evolution_timer = 0.0f;

    // Camera settings
    float rotation_x = 0.0f;
    float rotation_y = 0.0f;
    float voxel_size = 0.8f;
    bool show_grid = true;
    bool enable_lighting = true;

    std::cout << "System initialized with " << alive_cells << " alive cells" << std::endl;

    // Timing for delta time
    float deltaTime = 0.0f;
    float lastFrame = 0.0f;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glfwPollEvents();

        // Process keyboard input for mouse control
        processKeyboardInput(window);

        // Process camera input
        if (cameraControlEnabled) {
            renderer.processInput(window, deltaTime);
        }

        // Auto evolution
        if (auto_evolve) {
            evolution_timer += deltaTime;
            if (evolution_timer >= (1.0f / evolution_speed)) {
                evolution_timer = 0.0f;

                if (evolution_method == 0) {
                    system.evolve();
                }
                else if (evolution_method == 1) {
                    system.evolveParallel();
                }
                else if (evolution_method == 2) {
                    system.evolveGPU();
                }
                alive_cells = system.getAliveCells();
            }
        }

        // Get window size for proper aspect ratio
        int window_width, window_height;
        glfwGetFramebufferSize(window, &window_width, &window_height);

        // === 3D RENDERING ===
        if (show_3d_view) {
            renderer.setVoxelSize(voxel_size);
            renderer.setShowGrid(show_grid);
            renderer.setEnableLighting(enable_lighting);
            renderer.setRotation(rotation_x, rotation_y);

            renderer.render(system.getCurrentState(), window_width, window_height);
        }
        else {
            glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }

        // === IMGUI INTERFACE ===
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Main Control Panel
        ImGui::Begin("Performance Testing - CPU vs GPU vs 3D Visualization");

        ImGui::Text("System: RTX 3060 Ti + %d CPU threads", num_threads);
        ImGui::Separator();

        // Evolution method selection
        ImGui::Text("Evolution Method:");
        ImGui::RadioButton("CPU Serial", &evolution_method, 0);
        ImGui::SameLine();
        ImGui::RadioButton("CPU Parallel (OpenMP)", &evolution_method, 1);
        ImGui::SameLine();
        ImGui::RadioButton("GPU CUDA", &evolution_method, 2);

        ImGui::Separator();

        // Auto evolution controls
        ImGui::Checkbox("Auto Evolution", &auto_evolve);
        if (auto_evolve) {
            ImGui::SameLine();
            ImGui::SliderFloat("Speed (steps/sec)", &evolution_speed, 0.1f, 10.0f);
        }

        // Action buttons
        if (ImGui::Button("STEP")) {
            auto start = std::chrono::high_resolution_clock::now();
            double stepCPUTime = 0.0;

            if (evolution_method == 0) {
                system.evolve();
                auto end = std::chrono::high_resolution_clock::now();
                stepCPUTime = std::chrono::duration<float, std::milli>(end - start).count();
                cpu_serial_time = static_cast<float>(stepCPUTime);
            }
            else if (evolution_method == 1) {
                system.evolveParallel();
                auto end = std::chrono::high_resolution_clock::now();
                stepCPUTime = std::chrono::duration<float, std::milli>(end - start).count();
                cpu_parallel_time = static_cast<float>(stepCPUTime);
            }
            else if (evolution_method == 2) {
                try {
                    gpu_cuda_time = static_cast<float>(system.evolveGPU());
                    std::cout << "GPU evolution completed in " << gpu_cuda_time << "ms" << std::endl;
                }
                catch (const std::exception& e) {
                    std::cerr << "GPU Error: " << e.what() << std::endl;
                    ImGui::Text("GPU Error - check console");
                }
            }

            // Add to performance tracking
            perfStats.addSample(cpu_serial_time, cpu_parallel_time, gpu_cuda_time);

            // Calculate speedups
            if (cpu_serial_time > 0 && cpu_parallel_time > 0) {
                cpu_speedup = cpu_serial_time / cpu_parallel_time;
            }
            if (cpu_serial_time > 0 && gpu_cuda_time > 0) {
                gpu_speedup = cpu_serial_time / gpu_cuda_time;
            }

            alive_cells = system.getAliveCells();

            const char* methods[] = { "Serial", "Parallel", "GPU" };
            std::cout << methods[evolution_method] << " evolution step. Alive: " << alive_cells << std::endl;
        }

        ImGui::SameLine();
        if (ImGui::Button("RESET")) {
            system.randomInitialize(0.2f);
            alive_cells = system.getAliveCells();
            cpu_serial_time = 0.0f;
            cpu_parallel_time = 0.0f;
            gpu_cuda_time = 0.0f;
            cpu_speedup = 0.0f;
            gpu_speedup = 0.0f;
            std::cout << "Reset. Alive: " << alive_cells << std::endl;
        }

        ImGui::SameLine();
        if (ImGui::Button("BENCHMARK ALL")) {
            std::cout << "\nRunning comprehensive benchmark (CPU + GPU)..." << std::endl;
            try {
                system.compareAllMethods(20);
            }
            catch (const std::exception& e) {
                std::cerr << "Benchmark error: " << e.what() << std::endl;
            }
        }

        // NEW Performance Stats buttons
        ImGui::SameLine();
        if (ImGui::Button("PERFORMANCE REPORT")) {
            perfStats.printReport(grid_size);
        }

        if (ImGui::Button("CLEAR STATS")) {
            perfStats.clear();
            std::cout << "Performance statistics cleared." << std::endl;
        }

        // Grid size control
        if (ImGui::SliderInt("Grid Size", &grid_size, 32, 128)) {
            system = DynamicSystem3D(grid_size, grid_size, grid_size);
            system.randomInitialize(0.2f);
            alive_cells = system.getAliveCells();

            renderer.cleanup();
            renderer.initialize(grid_size, grid_size, grid_size);

            cpu_serial_time = 0.0f;
            cpu_parallel_time = 0.0f;
            gpu_cuda_time = 0.0f;
            cpu_speedup = 0.0f;
            gpu_speedup = 0.0f;
        }

        ImGui::Separator();

        // Performance display
        ImGui::Text("Performance Results:");
        if (cpu_serial_time > 0) {
            ImGui::Text("CPU Serial:     %.3f ms", cpu_serial_time);
        }
        if (cpu_parallel_time > 0) {
            ImGui::Text("CPU Parallel:   %.3f ms (%d threads)", cpu_parallel_time, num_threads);
        }
        if (gpu_cuda_time > 0) {
            ImGui::Text("GPU CUDA:       %.3f ms", gpu_cuda_time);
        }

        ImGui::Separator();

        // Speedup display
        ImGui::Text("Speedups:");
        if (cpu_speedup > 0) {
            ImGui::Text("CPU Parallel:   %.2fx vs Serial", cpu_speedup);
        }
        if (gpu_speedup > 0) {
            ImGui::Text("GPU CUDA:       %.1fx vs Serial", gpu_speedup);

            if (gpu_speedup > 100) {
                ImGui::TextColored(ImVec4(0, 1, 0, 1), "EXCELLENT GPU Performance!");
            }
            else if (gpu_speedup > 50) {
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "GOOD GPU Performance");
            }
            else if (gpu_speedup > 10) {
                ImGui::TextColored(ImVec4(1, 0.5, 0, 1), "Modest GPU speedup");
            }
        }

        ImGui::Separator();

        // NEW: Performance Statistics Display
        ImGui::Text("Performance Statistics:");
        ImGui::Text("Samples collected: %d", perfStats.sampleCount);
        if (perfStats.sampleCount > 0) {
            ImGui::Text("Average GPU speedup: %.1fx", perfStats.getAverageGPUSpeedup());
            ImGui::Text("Average CPU speedup: %.1fx", perfStats.getAverageCPUSpeedup());
            ImGui::Text("Best GPU speedup: %.1fx", perfStats.bestGPUSpeedup);
            ImGui::Text("GPU Throughput: %.2f M cells/sec", perfStats.getGPUThroughput(grid_size));
            ImGui::Text("CPU Efficiency: %.1f%%", perfStats.getCPUEfficiency());

            double avgSpeedup = perfStats.getAverageGPUSpeedup();
            if (avgSpeedup > 50) {
                ImGui::TextColored(ImVec4(0, 1, 0, 1), "Overall Rating: EXCELLENT");
            }
            else if (avgSpeedup > 20) {
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "Overall Rating: GOOD");
            }
            else if (avgSpeedup > 5) {
                ImGui::TextColored(ImVec4(1, 0.5, 0, 1), "Overall Rating: MODEST");
            }
            else {
                ImGui::TextColored(ImVec4(1, 0, 0, 1), "Overall Rating: NEEDS IMPROVEMENT");
            }
        }

        ImGui::Separator();

        // System status
        ImGui::Text("System Status:");
        ImGui::Text("Grid: %dx%dx%d (%d cells)", grid_size, grid_size, grid_size, grid_size * grid_size * grid_size);
        ImGui::Text("Alive Cells: %d", alive_cells);
        ImGui::Text("Rendered Voxels: %d", renderer.getLastRenderedVoxels());
        ImGui::Text("App FPS: %.1f", ImGui::GetIO().Framerate);

        ImGui::End();

        // === 3D VISUALIZATION CONTROLS ===
        ImGui::Begin("3D Visualization Controls");

        ImGui::Checkbox("Show 3D View", &show_3d_view);

        if (show_3d_view) {
            ImGui::Separator();

            // Camera controls
            ImGui::Text("Camera Controls:");

            if (cameraControlEnabled) {
                ImGui::TextColored(ImVec4(0, 1, 0, 1), "Mouse Look: ACTIVE");
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "WASD to move, Mouse to look, Q/E up/down");
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "Mouse wheel to zoom");
                ImGui::TextColored(ImVec4(1, 0.5, 0, 1), "Press TAB to disable mouse look");
            }
            else {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "Mouse Look: DISABLED");
                ImGui::TextColored(ImVec4(1, 1, 1, 1), "Press TAB to enable mouse look");
            }

            if (mouseLookToggled) {
                ImGui::Separator();
                if (cameraControlEnabled) {
                    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Mouse look enabled! Use WASD to move");
                }
                else {
                    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Mouse look disabled");
                }

                static float toggleTimer = 0.0f;
                toggleTimer += deltaTime;
                if (toggleTimer > 2.0f) {
                    mouseLookToggled = false;
                    toggleTimer = 0.0f;
                }
            }

            // Manual rotation
            ImGui::Text("Manual Rotation:");
            ImGui::SliderFloat("X Rotation", &rotation_x, -180.0f, 180.0f);
            ImGui::SliderFloat("Y Rotation", &rotation_y, -180.0f, 180.0f);

            ImGui::Separator();

            // Visual settings
            ImGui::Text("Visual Settings:");
            ImGui::SliderFloat("Voxel Size", &voxel_size, 0.1f, 1.0f);
            ImGui::Checkbox("Show Grid", &show_grid);
            ImGui::Checkbox("Enable Lighting", &enable_lighting);

            ImGui::Separator();

            // Camera info
            ImGui::Text("Camera Info:");
            glm::vec3 camPos = renderer.getCameraPosition();
            ImGui::Text("Position: (%.1f, %.1f, %.1f)", camPos.x, camPos.y, camPos.z);

            ImGui::Separator();

            // Performance info
            ImGui::Text("Rendering Performance:");
            ImGui::Text("Voxels rendered: %d / %d", renderer.getLastRenderedVoxels(),
                grid_size * grid_size * grid_size);
            float density = static_cast<float>(renderer.getLastRenderedVoxels()) / (grid_size * grid_size * grid_size) * 100.0f;
            ImGui::Text("Population density: %.1f%%", density);
        }

        ImGui::End();

        // === IMPLEMENTATION STATUS ===
        ImGui::Begin("Implementation Progress + Performance Analysis");

        ImGui::TextColored(ImVec4(0, 1, 0, 1), "[COMPLETE] Visual Studio 2022 + CUDA 13");
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "[COMPLETE] OpenGL + vcpkg libraries");
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "[COMPLETE] Project structure created");
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "[COMPLETE] DynamicSystem3D implementation");
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "[COMPLETE] CPU parallel (OpenMP)");
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "[COMPLETE] GPU CUDA kernels");
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "[COMPLETE] 3D visualization with OpenGL");
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "[COMPLETE] Performance statistics tracking");

        ImGui::Separator();

        // Feature list
        ImGui::Text("Implemented Features:");
        ImGui::BulletText("Real-time 3D voxel rendering");
        ImGui::BulletText("Interactive camera controls (TAB toggle)");
        ImGui::BulletText("Performance comparison tools");
        ImGui::BulletText("Advanced performance statistics");
        ImGui::BulletText("Auto-evolution with speed control");
        ImGui::BulletText("Dynamic grid resizing");
        ImGui::BulletText("Lighting and visual effects");
        ImGui::BulletText("Population density visualization");
        ImGui::BulletText("Comprehensive benchmarking suite");

        ImGui::Separator();

        // Controls
        ImGui::Text("Controls:");
        ImGui::BulletText("Press TAB to toggle mouse look camera");
        ImGui::BulletText("When mouse look active: WASD + mouse to navigate");
        ImGui::BulletText("Use STEP to test individual methods");
        ImGui::BulletText("Use BENCHMARK ALL for complete comparison");
        ImGui::BulletText("Use PERFORMANCE REPORT for detailed analysis");
        ImGui::BulletText("Larger grids show bigger GPU advantages");

        ImGui::End();

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    renderer.cleanup();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();

    std::cout << "Application shutting down..." << std::endl;
    return 0;
}