// 三缓冲区 Graph
class GraphBasedPipeline
{
    struct Frame
    {
        GLuint pbo;
        cudaGraphicsResource_t cudaRes;
        cudaStream_t stream;
        cudaGraphExec_t graphExec;
        uchar4 *devPtr; // 持久指针
    };

    Frame frames[3];
    cudaGraph_t capturedGraph;

    void captureGraphOnce()
    {
        // 在捕获前，先设置好资源
        for (auto &frame : frames)
        {
            cudaGraphicsMapResources(1, &frame.cudaRes, frame.stream);
            cudaGraphicsResourceGetMappedPointer((void **)&frame.devPtr,
                                                 nullptr, frame.cudaRes);
            cudaGraphicsUnmapResources(1, &frame.cudaRes, frame.stream);
        }

        // 开始捕获（使用第一个流的配置）
        cudaStreamBeginCapture(frames[0].stream, cudaStreamCaptureModeGlobal);

        // 录制典型操作
        cudaGraphicsMapResources(1, &frames[0].cudaRes, frames[0].stream);

        // 使用持久指针
        randomKernel<<<grid, block, 0, frames[0].stream>>>(
            frames[0].devPtr, width, height, 0 // 0是占位符
        );

        // 这里不会阻塞！只是记录同步点
        cudaGraphicsUnmapResources(1, &frames[0].cudaRes, frames[0].stream);

        cudaStreamEndCapture(frames[0].stream, &capturedGraph);

        // 为每个帧实例化图
        for (int i = 0; i < 3; i++)
        {
            // 配置更新：使用对应帧的资源和指针
            cudaGraphInstantiate(&frames[i].graphExec, capturedGraph,
                                 updateNodeParams, &frames[i], 0);
        }
    }

    static void updateNodeParams(cudaGraphNode_t node, void *userData)
    {
        Frame *frame = (Frame *)userData;
        cudaGraphNodeType type;
        cudaGraphNodeGetType(node, &type);

        if (type == cudaGraphNodeTypeKernel)
        {
            // 更新 kernel 参数，使用当前帧的 devPtr
            cudaKernelNodeParams params;
            cudaGraphNodeGetParams(node, &params);

            // 修改指针参数
            void *kernelArgs[4];
            memcpy(kernelArgs, params.kernelParams, 4 * sizeof(void *));
            kernelArgs[0] = &frame->devPtr; // 第一个参数：数据指针

            params.kernelParams = kernelArgs;
            cudaGraphNodeSetParams(node, &params);
        }
    }
};