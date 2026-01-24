#include <vector>
#include <iostream>
#include <cuda_runtime.h>
// Graph 化的三缓冲区
class GraphPipeline
{
private:
    static constexpr int NUM_BUFFERS = 3;

    struct Buffer
    {
        GLuint pbo;
        GLuint texture;
        cudaGraphicsResource_t cudaResource;
        cudaStream_t stream;
        cudaEvent_t readyEvent;
        cudaGraphExec_t graphExec;
    };

    std::vector<Buffer> buffers;
    cudaGraph_t captureGraph;
    bool graphCaptured = false;
    int frameCounter = 0;

    // 我们的 CUDA kernel
    __global__ void generatePattern(uchar4 *data, int width, int height, int frame)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height)
        {
            int idx = y * width + x;
            float fx = (float)x / width;
            float fy = (float)y / height;

            // 生成动态图案
            float time = frame * 0.01f;
            float r = 0.5f + 0.5f * sinf(fx * 10.0f + time);
            float g = 0.5f + 0.5f * cosf(fy * 10.0f + time * 1.3f);
            float b = 0.5f + 0.5f * sinf((fx + fy) * 5.0f + time * 0.7f);

            data[idx].x = (unsigned char)(r * 255);
            data[idx].y = (unsigned char)(g * 255);
            data[idx].z = (unsigned char)(b * 255);
            data[idx].w = 255;
        }
    }

public:
    GraphPipeline(int width, int height) : width(width), height(height)
    {
        buffers.resize(NUM_BUFFERS);

        // 初始化每个缓冲区
        for (int i = 0; i < NUM_BUFFERS; i++)
        {
            Buffer &buf = buffers[i];

            // 创建 OpenGL 资源
            glGenBuffers(1, &buf.pbo);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf.pbo);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4,
                         nullptr, GL_STREAM_DRAW);

            glGenTextures(1, &buf.texture);
            glBindTexture(GL_TEXTURE_2D, buf.texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

            // 创建 CUDA 资源
            cudaStreamCreate(&buf.stream);
            cudaEventCreate(&buf.readyEvent, cudaEventDisableTiming);
            cudaGraphicsGLRegisterBuffer(&buf.cudaResource, buf.pbo,
                                         cudaGraphicsRegisterFlagsWriteDiscard);
        }

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    ~GraphPipeline()
    {
        for (auto &buf : buffers)
        {
            cudaEventDestroy(buf.readyEvent);
            cudaStreamDestroy(buf.stream);
            cudaGraphicsUnregisterResource(buf.cudaResource);
            glDeleteBuffers(1, &buf.pbo);
            glDeleteTextures(1, &buf.texture);

            if (buf.graphExec)
            {
                cudaGraphExecDestroy(buf.graphExec);
            }
        }

        if (graphCaptured)
        {
            cudaGraphDestroy(captureGraph);
        }
    }

    // 捕获计算 Graph（只在初始化时调用一次）
    void captureComputeGraph()
    {
        if (graphCaptured)
            return;

        // 使用第一个 buffer 的流进行捕获
        cudaStream_t captureStream = buffers[0].stream;

        // 开始捕获
        cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal);

        // 录制典型操作序列
        cudaGraphicsResource_t dummyResource = buffers[0].cudaResource;
        cudaGraphicsMapResources(1, &dummyResource, captureStream);

        uchar4 *dummyPtr;
        size_t dummySize;
        cudaGraphicsResourceGetMappedPointer((void **)&dummyPtr, &dummySize,
                                             dummyResource);

        // 录制 kernel 启动（参数使用占位符）
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);

        int dummyFrame = 0; // 占位符，后面会更新
        generatePattern<<<grid, block, 0, captureStream>>>(dummyPtr, width,
                                                           height, dummyFrame);

        // 录制事件记录
        cudaEvent_t dummyEvent = buffers[0].readyEvent;
        cudaEventRecord(dummyEvent, captureStream);

        cudaGraphicsUnmapResources(1, &dummyResource, captureStream);

        // 结束捕获
        cudaStreamEndCapture(captureStream, &captureGraph);

        // 为每个 buffer 实例化独立的 Graph
        for (int i = 0; i < NUM_BUFFERS; i++)
        {
            // 创建实例化时的更新配置
            struct UpdateConfig
            {
                cudaGraphicsResource_t *resource;
                cudaEvent_t *event;
                int bufferIndex;
            };

            UpdateConfig config = {&buffers[i].cudaResource,
                                   &buffers[i].readyEvent, i};

            // 实例化，允许后续参数更新
            cudaGraphInstantiate(&buffers[i].graphExec, captureGraph,
                                 nullptr, &config, 0);
        }

        graphCaptured = true;
        std::cout << "CUDA Graph 捕获完成，每个缓冲区都有独立的实例" << std::endl;
    }

    // 渲染一帧
    void renderFrame()
    {
        if (!graphCaptured)
        {
            captureComputeGraph();
        }

        int computeIdx = frameCounter % NUM_BUFFERS;
        int uploadIdx = (frameCounter + 1) % NUM_BUFFERS;
        int displayIdx = (frameCounter + 2) % NUM_BUFFERS;

        // ========== 阶段1：Graph 执行计算 ==========
        Buffer &computeBuf = buffers[computeIdx];

        // 更新 Graph 参数（帧号）
        cudaGraphNode_t *nodes;
        size_t numNodes;
        cudaGraphGetNodes(captureGraph, &nodes, &numNodes);

        for (size_t i = 0; i < numNodes; i++)
        {
            cudaGraphNodeType type;
            cudaGraphNodeGetType(nodes[i], &type);

            if (type == cudaGraphNodeTypeKernel)
            {
                // 更新 kernel 参数
                cudaKernelNodeParams params;
                cudaGraphNodeGetParams(nodes[i], &params);

                // 修改帧号参数
                void *kernelArgs[4];
                memcpy(kernelArgs, params.kernelParams, 4 * sizeof(void *));

                static int frameParam = frameCounter;
                kernelArgs[3] = &frameParam; // 第四个参数是帧号

                params.kernelParams = kernelArgs;
                cudaGraphExecKernelNodeSetParams(computeBuf.graphExec,
                                                 nodes[i], &params);
                break;
            }
        }

        // 执行 Graph（一次性提交所有操作！）
        cudaGraphLaunch(computeBuf.graphExec, computeBuf.stream);

        // ========== 阶段2：上传前一帧数据 ==========
        if (frameCounter >= 1)
        {
            Buffer &uploadBuf = buffers[uploadIdx];

            // 等待计算完成
            cudaEventSynchronize(uploadBuf.readyEvent);

            // 上传到纹理
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, uploadBuf.pbo);
            glBindTexture(GL_TEXTURE_2D, uploadBuf.texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                            GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        }

        // ========== 阶段3：渲染更前一帧 ==========
        if (frameCounter >= 2)
        {
            Buffer &displayBuf = buffers[displayIdx];

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, displayBuf.texture);
            // ... 渲染全屏四边形
        }

        frameCounter++;
    }

private:
    int width, height;
};