#ifndef RENDERER3D_H
#define RENDERER3D_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>  // ADD GLFW INCLUDE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <string>
#include <algorithm>  // ADD FOR std::max

class Renderer3D {
private:
    // OpenGL objects
    GLuint shaderProgram;
    GLuint VAO, VBO, EBO;
    GLuint instanceVBO; // For instanced rendering

    // Shader uniform locations
    GLint mvpLocation;
    GLint colorLocation;
    GLint cameraPositionLocation;
    GLint lightPositionLocation;

    // Camera
    glm::vec3 cameraPos;
    glm::vec3 cameraFront;
    glm::vec3 cameraUp;
    float yaw, pitch;
    float fov;

    // Grid dimensions
    int gridWidth, gridHeight, gridDepth;

    // Rendering settings
    float voxelSize;
    bool showGrid;
    bool enableLighting;
    float rotationX, rotationY;

    // Performance
    int lastRenderedVoxels;

public:
    Renderer3D();
    ~Renderer3D();

    // Initialization
    bool initialize(int width, int height, int depth);
    void cleanup();

    // Rendering
    void render(const std::vector<int>& grid, int windowWidth, int windowHeight);
    void renderGrid();
    void renderVoxels(const std::vector<int>& grid);

    // Camera controls
    void processInput(GLFWwindow* window, float deltaTime);
    void processMouseMovement(float xOffset, float yOffset);
    void processMouseScroll(float yOffset);

    // Settings
    void setVoxelSize(float size) { voxelSize = size; }
    void setShowGrid(bool show) { showGrid = show; }
    void setEnableLighting(bool enable) { enableLighting = enable; }
    void setRotation(float x, float y) { rotationX = x; rotationY = y; }

    // Info
    int getLastRenderedVoxels() const { return lastRenderedVoxels; }
    glm::vec3 getCameraPosition() const { return cameraPos; }

private:
    // Shader management
    bool loadShaders();
    GLuint compileShader(const std::string& source, GLenum type);
    GLuint linkProgram(GLuint vertexShader, GLuint fragmentShader);

    // Geometry setup
    void setupCubeGeometry();
    void setupInstancedRendering();

    // Matrix calculations
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix(float aspectRatio) const;

    // Shader sources - moved to private static
    static const char* getVertexShaderSource();
    static const char* getFragmentShaderSource();
};

#endif // RENDERER3D_H