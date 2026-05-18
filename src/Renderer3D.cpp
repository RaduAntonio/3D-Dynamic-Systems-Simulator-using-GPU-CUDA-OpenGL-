#include "Renderer3D.h"
#include <iostream>
#include <GLFW/glfw3.h>

// Shader sources - now in implementation file
const char* Renderer3D::getVertexShaderSource() {
    return R"(
#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aInstancePos;

uniform mat4 mvp;
uniform vec3 cameraPosition;
uniform vec3 lightPosition;

out vec3 FragPos;
out vec3 Normal;
out vec3 LightPos;
out vec3 ViewPos;

void main() {
    vec3 worldPos = aPos + aInstancePos;
    gl_Position = mvp * vec4(worldPos, 1.0);
    
    FragPos = worldPos;
    Normal = aNormal;
    LightPos = lightPosition;
    ViewPos = cameraPosition;
}
)";
}

const char* Renderer3D::getFragmentShaderSource() {
    return R"(
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec3 LightPos;
in vec3 ViewPos;

uniform vec3 color;

out vec4 FragColor;

void main() {
    // Ambient lighting
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * color;
    
    // Diffuse lighting
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(LightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * color;
    
    // Specular lighting
    float specularStrength = 0.5;
    vec3 viewDir = normalize(ViewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
    
    // Combine lighting
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 0.8); // Slightly transparent
}
)";
}

Renderer3D::Renderer3D()
    : shaderProgram(0), VAO(0), VBO(0), EBO(0), instanceVBO(0),
    cameraPos(0.0f, 0.0f, 100.0f),
    cameraFront(0.0f, 0.0f, -1.0f),
    cameraUp(0.0f, 1.0f, 0.0f),
    yaw(-90.0f), pitch(0.0f), fov(45.0f),
    voxelSize(0.8f), showGrid(true), enableLighting(true),
    rotationX(0.0f), rotationY(0.0f), lastRenderedVoxels(0) {
}

Renderer3D::~Renderer3D() {
    cleanup();
}

bool Renderer3D::initialize(int width, int height, int depth) {
    gridWidth = width;
    gridHeight = height;
    gridDepth = depth;

    // Load and compile shaders
    if (!loadShaders()) {
        std::cerr << "Failed to load shaders" << std::endl;
        return false;
    }

    // Setup cube geometry
    setupCubeGeometry();

    // Setup instanced rendering
    setupInstancedRendering();

    // OpenGL settings
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_CULL_FACE);

    // Center camera on grid
    float centerX = width * 0.5f;
    float centerY = height * 0.5f;
    float centerZ = depth * 0.5f;
    float maxDim = std::max({ width, height, depth });
    cameraPos = glm::vec3(centerX, centerY, centerZ + maxDim * 1.5f);

    std::cout << "3D Renderer initialized for " << width << "x" << height << "x" << depth << " grid" << std::endl;
    return true;
}

void Renderer3D::cleanup() {
    if (shaderProgram) glDeleteProgram(shaderProgram);
    if (VAO) glDeleteVertexArrays(1, &VAO);
    if (VBO) glDeleteBuffers(1, &VBO);
    if (EBO) glDeleteBuffers(1, &EBO);
    if (instanceVBO) glDeleteBuffers(1, &instanceVBO);
}

void Renderer3D::render(const std::vector<int>& grid, int windowWidth, int windowHeight) {
    // Clear screen
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Setup viewport
    glViewport(0, 0, windowWidth, windowHeight);

    // Use shader program
    glUseProgram(shaderProgram);

    // Calculate matrices
    glm::mat4 model = glm::mat4(1.0f);

    // Apply rotations
    model = glm::rotate(model, glm::radians(rotationX), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(rotationY), glm::vec3(0.0f, 1.0f, 0.0f));

    // Center the grid
    float centerX = gridWidth * 0.5f;
    float centerY = gridHeight * 0.5f;
    float centerZ = gridDepth * 0.5f;
    model = glm::translate(model, glm::vec3(-centerX, -centerY, -centerZ));

    glm::mat4 view = getViewMatrix();
    glm::mat4 projection = getProjectionMatrix((float)windowWidth / windowHeight);
    glm::mat4 mvp = projection * view * model;

    // Set uniforms
    glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform3fv(cameraPositionLocation, 1, glm::value_ptr(cameraPos));

    // Light position (follows camera)
    glm::vec3 lightPos = cameraPos + glm::vec3(10.0f, 10.0f, 10.0f);
    glUniform3fv(lightPositionLocation, 1, glm::value_ptr(lightPos));

    // Render grid lines (optional)
    if (showGrid) {
        renderGrid();
    }

    // Render voxels
    renderVoxels(grid);
}

void Renderer3D::renderGrid() {
    // Simple grid rendering - just the boundaries
    glUniform3f(colorLocation, 0.3f, 0.3f, 0.3f); // Dark gray

    // This would render wireframe grid - simplified for now
    // Could add line rendering here if needed
}

void Renderer3D::renderVoxels(const std::vector<int>& grid) {
    // Prepare instance positions for alive cells
    std::vector<glm::vec3> instancePositions;
    instancePositions.reserve(grid.size()); // Reserve max possible

    for (int z = 0; z < gridDepth; z++) {
        for (int y = 0; y < gridHeight; y++) {
            for (int x = 0; x < gridWidth; x++) {
                int index = x + y * gridWidth + z * gridWidth * gridHeight;
                if (grid[index] == 1) { // Alive cell
                    instancePositions.push_back(glm::vec3(x, y, z));
                }
            }
        }
    }

    lastRenderedVoxels = instancePositions.size();

    if (instancePositions.empty()) return;

    // Update instance buffer
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, instancePositions.size() * sizeof(glm::vec3),
        instancePositions.data(), GL_DYNAMIC_DRAW);

    // Set voxel color based on density
    float density = (float)instancePositions.size() / (gridWidth * gridHeight * gridDepth);
    glm::vec3 voxelColor;

    if (density > 0.5f) {
        voxelColor = glm::vec3(1.0f, 0.2f, 0.2f); // Red - high density
    }
    else if (density > 0.2f) {
        voxelColor = glm::vec3(0.2f, 1.0f, 0.2f); // Green - medium density
    }
    else {
        voxelColor = glm::vec3(0.2f, 0.2f, 1.0f); // Blue - low density
    }

    glUniform3fv(colorLocation, 1, glm::value_ptr(voxelColor));

    // Render instanced cubes
    glBindVertexArray(VAO);
    glDrawElementsInstanced(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0, instancePositions.size());
    glBindVertexArray(0);
}

void Renderer3D::processInput(GLFWwindow* window, float deltaTime) {
    float cameraSpeed = 50.0f * deltaTime;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraUp;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraUp;
}

void Renderer3D::processMouseMovement(float xOffset, float yOffset) {
    float sensitivity = 0.1f;
    xOffset *= sensitivity;
    yOffset *= sensitivity;

    yaw += xOffset;
    pitch += yOffset;

    // Constrain pitch
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    // Calculate new front vector
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

void Renderer3D::processMouseScroll(float yOffset) {
    fov -= yOffset;
    if (fov < 1.0f) fov = 1.0f;
    if (fov > 45.0f) fov = 45.0f;
}

bool Renderer3D::loadShaders() {
    GLuint vertexShader = compileShader(getVertexShaderSource(), GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(getFragmentShaderSource(), GL_FRAGMENT_SHADER);

    if (vertexShader == 0 || fragmentShader == 0) {
        return false;
    }

    shaderProgram = linkProgram(vertexShader, fragmentShader);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    if (shaderProgram == 0) {
        return false;
    }

    // Get uniform locations
    mvpLocation = glGetUniformLocation(shaderProgram, "mvp");
    colorLocation = glGetUniformLocation(shaderProgram, "color");
    cameraPositionLocation = glGetUniformLocation(shaderProgram, "cameraPosition");
    lightPositionLocation = glGetUniformLocation(shaderProgram, "lightPosition");

    return true;
}

GLuint Renderer3D::compileShader(const std::string& source, GLenum type) {
    GLuint shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
        return 0;
    }

    return shader;
}

GLuint Renderer3D::linkProgram(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "Program linking failed: " << infoLog << std::endl;
        return 0;
    }

    return program;
}

void Renderer3D::setupCubeGeometry() {
    // Complete cube vertices with normals (all 6 faces)
    float vertices[] = {
        // Back face
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,

        // Front face
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,

        // Left face
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,

        // Right face
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,

         // Bottom face
         -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
          0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
          0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
         -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,

         // Top face
         -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
          0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
          0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
         -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f
    };

    unsigned int indices[] = {
        0,  1,  2,   2,  3,  0,   // Back face
        4,  5,  6,   6,  7,  4,   // Front face
        8,  9,  10,  10, 11, 8,   // Left face
        12, 13, 14,  14, 15, 12,  // Right face
        16, 17, 18,  18, 19, 16,  // Bottom face
        20, 21, 22,  22, 23, 20   // Top face
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void Renderer3D::setupInstancedRendering() {
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);

    // Instance position attribute
    glBindVertexArray(VAO);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1); // One per instance

    glBindVertexArray(0);
}

glm::mat4 Renderer3D::getViewMatrix() const {
    return glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
}

glm::mat4 Renderer3D::getProjectionMatrix(float aspectRatio) const {
    return glm::perspective(glm::radians(fov), aspectRatio, 0.1f, 1000.0f);
}