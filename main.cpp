#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thread>
#include <iostream>
#include <vector>
#include <string>
constexpr int WIDTH = 1024;
constexpr int HEIGHT = 1024;
/* ================= Fullscreen Quad Shader ================= */
/**
 * @brief 顶点着色器源码
    输出：纹理坐标uv
    创建全屏四边形：4个顶点覆盖整个屏幕
    gl_VertexID：顶点的内置索引(0-3)
    uv坐标：从[-1,1]转换到[0,1]
 *
 */
const char *kVS = R"(#version 450 core
out vec2 uv;
void main() {
    vec2 pos[4] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0,  1.0)
    );
    uv = pos[gl_VertexID] * 0.5 + 0.5;
    gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
}
)";
/**
 * @brief 片段着色器源码
        输入：纹理坐标uv
        输出：像素颜色outColor
        功能：从纹理采样颜色
 */
const char *kFS = R"(#version 450 core
in vec2 uv;
out vec4 outColor;
uniform sampler2D uTex;
void main() {
    outColor = texture(uTex, uv);
}
)";
constexpr int NUM_PBO = 3;
GLFWwindow *window;
// --- FPS + 延迟统计 ---
struct fps_count
{
    int frames = 0;
    float fps = 0.0f;
    double gpuTimeMs = 0.0;
    std::chrono::high_resolution_clock::time_point lastTime = std::chrono::high_resolution_clock::now();
};
fps_count stats;
// 状态
enum class frame_state : uint8_t
{
    FREE,         //  空闲，可以给 CUDA 写
    CUDA_RUNNING, // CUDA 计算中
    CUDA_DONE,    // CUDA 计算完成，数据就绪
    GL_UPLOADING, // 正在上传到纹理
    GL_RENDERING  // 正在渲染
};
/* ================= CUDA Kernel ================= */
extern void launch_cuda_kernel(uchar4 *devPtr, int w, int h, cudaStream_t stream);
struct frame
{
    size_t id;
    GLuint pbo;
    GLuint tex;
    cudaGraphicsResource *cudaPBO;
    cudaStream_t stream;
    cudaGraph_t graphs;
    cudaEvent_t cudaDone;
    cudaGraphExec_t graphExec;
    uchar4 *devPtr;
    size_t size;
    GLsync glFence{0}; // GL finished reading
    frame_state state{frame_state::FREE};

    std::chrono::high_resolution_clock::time_point startTime;
    double latencyMs = 0.0;
};
frame pipleLine[NUM_PBO];
void initFrame()
{

    /* ---------- PBO(Pixel Buffer Object) ---------- */

    for (int i = 0; i < NUM_PBO; ++i)
    {
        pipleLine[i].id = i;
        /* ---------- PBO(Pixel Buffer Object) ---------- */
        glGenBuffers(1, &pipleLine[i].pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pipleLine[i].pbo);                             // 绑定为像素解包缓冲区
        glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 4, nullptr, GL_DYNAMIC_DRAW); // 分配内存

        /* ---------- Texture ---------- */
        glGenTextures(1, &pipleLine[i].tex);
        glBindTexture(GL_TEXTURE_2D, pipleLine[i].tex);                                                 // 绑定为2D纹理
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr); // 分配纹理内存（RGBA8格式，800x600）
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);                              // 缩小过滤
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);                              // 放大过滤

        /* ---------- CUDA resource ---------- */
        cudaStreamCreate(&pipleLine[i].stream);                                                                       // 创建cudal流
        cudaEventCreate(&pipleLine[i].cudaDone, cudaEventDisableTiming);                                              // 创建Event
        cudaGraphicsGLRegisterBuffer(&pipleLine[i].cudaPBO, pipleLine[i].pbo, cudaGraphicsRegisterFlagsWriteDiscard); // 注册PBO为CUDA资源，允许CUDA直接写入
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0); 
}

void captureComputeGraph()
{
    for (int i = 0; i < NUM_PBO; ++i)
    {
        // 清理旧的 Graph
        if (pipleLine[i].graphExec)
            cudaGraphExecDestroy(pipleLine[i].graphExec);
        if (pipleLine[i].graphs)
            cudaGraphDestroy(pipleLine[i].graphs);
        // --- Map ---
        cudaGraphicsMapResources(1, &pipleLine[i].cudaPBO, pipleLine[i].stream);                                       // 资源映射, 阻塞，等待之前的所有CUDA和OpenGL操作完成
        cudaStreamBeginCapture(pipleLine[i].stream, cudaStreamCaptureModeGlobal);                                      // 开始录制操作序列（不实际执行，只记录）
        cudaGraphicsResourceGetMappedPointer((void **)&pipleLine[i].devPtr, &pipleLine[i].size, pipleLine[i].cudaPBO); // 获取设备指针
        // --- Kernel ---
        launch_cuda_kernel(pipleLine[i].devPtr, WIDTH, HEIGHT, pipleLine[i].stream);             // 执行CUDA核函数
        cudaStreamEndCapture(pipleLine[i].stream, &pipleLine[i].graphs);                         // 结束捕获
        cudaGraphInstantiate(&pipleLine[i].graphExec, pipleLine[i].graphs, nullptr, nullptr, 0); // 实例化阶段
        // --- Unmap ---
        cudaGraphicsUnmapResources(1, &pipleLine[i].cudaPBO, pipleLine[i].stream); // 取消映射 等待映射期间的所有CUDA操作完成
    }
}

void destroyFrame()
{
    /* ---------- Cleanup ---------- */
    for (int i = 0; i < NUM_PBO; ++i)
    {
        if (pipleLine[i].glFence)
        {
            glDeleteSync(pipleLine[i].glFence);
            pipleLine[i].glFence = 0;
        }
        cudaStreamSynchronize(pipleLine[i].stream);
        cudaEventDestroy(pipleLine[i].cudaDone);
        cudaStreamDestroy(pipleLine[i].stream);
        cudaGraphicsUnregisterResource(pipleLine[i].cudaPBO); // 接触图形互操作资源，释放相关CUDA资源
        glDeleteBuffers(1, &pipleLine[i].pbo);
        glDeleteTextures(1, &pipleLine[i].tex);
        cudaGraphExecDestroy(pipleLine[i].graphExec);
        cudaGraphDestroy(pipleLine[i].graphs);
    }
}
/* ================= Shader Utils ================= */
/**
 * @brief 编译着色器
 *
 * @param type
            GL_VERTEX_SHADER	顶点着色器
            GL_FRAGMENT_SHADER	片元着色器（像素着色器）
            GL_GEOMETRY_SHADER	几何着色器
            GL_TESS_CONTROL_SHADER	细分控制着色器
            GL_TESS_EVALUATION_SHADER	细分计算着色器
            GL_COMPUTE_SHADER	计算着色器
 * @param src 着色器源代码
 * @return GLuint
 */
GLuint compileShader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);     // 创建着色器对象
    glShaderSource(s, 1, &src, nullptr); // 设置着色器源代码
    glCompileShader(s);                  // 编译着色器

    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetShaderInfoLog(s, 1024, nullptr, log); // 获取错误日志
        std::cerr << "Shader compile error:\n"
                  << log << std::endl; // 输出错误
    }
    return s;
}
/**
 * @brief 创建着色器程序
 *
 * @param vs
 * @param fs
 * @return GLuint
 */
GLuint createProgram(const char *vs, const char *fs)
{
    GLuint p = glCreateProgram();                     // 创建程序对象
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);   // 编译顶点着色器
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs); // 编译片段着色器
    glAttachShader(p, v);                             // 附加顶点着色器到程序
    glAttachShader(p, f);                             // 附加片段着色器到程序
    glLinkProgram(p);                                 // 链接着色器程序

    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok); // 检查链接状态
    if (!ok)
    {
        char log[1024];
        glGetProgramInfoLog(p, 1024, nullptr, log);
        std::cerr << "Program link error:\n"
                  << log << std::endl;
    }
    glDeleteShader(v); // 删除着色器对象（已链接到程序）
    glDeleteShader(f); // 删除着色器对象
    return p;
}
int create_window()
{
    if (!glfwInit())
        return -1;
    // 设置OpenGL版本和配置
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);                 // OpenGL主版本4
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);                 // OpenGL次版本4.5
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 核心模式
    // 创建窗口
    window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA + OpenGL Interop (Modern)", nullptr, nullptr); // 核心模式
    glfwMakeContextCurrent(window);                                                               // 设置为当前上下文
                                                                                                  // 初始化GLAD（加载OpenGL函数指针）
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))                                      // 初始化GLAD（加载OpenGL函数指针）
    {
        std::cerr << "Failed to init GLAD\n";
        return -1;
    }
    return 0;
}

int main()
{

    if (create_window() == -1)
    {
        std::cout << "OpenGL init: " << glGetString(GL_VERSION) << std::endl;
        return -1;
    }
    initFrame();
    captureComputeGraph();
    /* ---------- Shader ---------- */
    GLuint prog = createProgram(kVS, kFS);              // 创建着色器程序
    glUseProgram(prog);                                 // 使用该程序
    glUniform1i(glGetUniformLocation(prog, "uTex"), 0); // 设置纹理单元为0

    /* ---------- VAO (empty Vertex Array Object) ---------- */
    GLuint vao;
    glGenVertexArrays(1, &vao); // 生成VAO（用于存储顶点状态）
    glBindVertexArray(vao);     // 绑定VAO
    int index = 0;
    /* ---------- Loop ---------- */
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    double lastTime = glfwGetTime();
    int frameCount = 0;
    while (!glfwWindowShouldClose(window) && frameCount < 100000)
    {
        start = std::chrono::high_resolution_clock::now();
        // 1. 找到可用的帧进行计算
        for (auto &frame : pipleLine)
        {
            if (frame.state == frame_state::FREE)
            {
                // 启动 CUDA 计算
                frame.startTime = std::chrono::high_resolution_clock::now();
                cudaGraphLaunch(frame.graphExec, frame.stream);
                cudaEventRecord(frame.cudaDone, frame.stream);
                frame.state = frame_state::CUDA_RUNNING;
            }
        }
        // 2. 检查 CUDA 是否完成
        for (auto &frame : pipleLine)
        {
            if (frame.state == frame_state::CUDA_RUNNING)
            {
                if (cudaEventQuery(frame.cudaDone) == cudaSuccess)
                {
                    frame.state = frame_state::CUDA_DONE;
                }
            }
        }
        for (auto &frame : pipleLine)
        {
            if (frame.state == frame_state::CUDA_DONE)
            {
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, frame.pbo);
                glBindTexture(GL_TEXTURE_2D, frame.tex);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                // 创建 GL 栅栏
                frame.glFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
                frame.state = frame_state::GL_UPLOADING;
            }
            if (frame.state == frame_state::GL_UPLOADING)
            {
                GLenum waitResult = glClientWaitSync(frame.glFence, 0, 0);
                if (waitResult == GL_ALREADY_SIGNALED || waitResult == GL_CONDITION_SATISFIED)
                {
                    // 渲染
                    glDeleteSync(frame.glFence);
                    glClear(GL_COLOR_BUFFER_BIT);
                    glUseProgram(prog);
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, frame.tex);
                    glBindVertexArray(vao);
                    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
                    glBindTexture(GL_TEXTURE_2D, 0);
                    glBindVertexArray(0);
                    glfwSwapBuffers(window);
                    glfwPollEvents();

                    auto t1 = std::chrono::high_resolution_clock::now();
                    frame.latencyMs = std::chrono::duration<double, std::milli>(t1 - frame.startTime).count();
                    frame.state = frame_state::FREE;
                    // FPS计数
                    stats.frames++;
                    auto now = std::chrono::high_resolution_clock::now();
                    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - stats.lastTime).count();
                    if (dt > 1000)
                    {
                        stats.fps = stats.frames * 1000.0f / dt;
                        stats.frames = 0;
                        stats.lastTime = now;
                        std::cout << "FPS: " << stats.fps << "\n";
                    }
                }
            }
        }
        frameCount++;
    }
    destroyFrame();
    glDeleteProgram(prog);         // 删除着色器程序
    glDeleteVertexArrays(1, &vao); // 删除VAO
    glfwDestroyWindow(window);     // 删除window
    glfwTerminate();               // 释放/删除分配的所有资源
}
