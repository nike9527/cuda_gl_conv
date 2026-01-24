// 终极优化：Graph + 事件 + 三缓冲区/高级技巧：Graph + 事件 + 流水线
class UltimatePipeline
{
    enum Stage
    {
        CPU_PREP,
        GPU_PROCESS,
        CPU_POST
    };

    struct WorkItem
    {
        Stage currentStage;
        cudaEvent_t gpuDone;
        void *data;
    };

    std::queue<WorkItem> workQueue;
    cudaGraphExec_t gpuGraph;
    cudaStream_t stream;

    void process()
    {
        // 1. 如果有待处理的CPU准备工作，做它
        if (!workQueue.empty() && workQueue.front().currentStage == CPU_PREP)
        {
            WorkItem &item = workQueue.front();
            cpuPrepare(item.data);
            item.currentStage = GPU_PROCESS;

            // 启动GPU处理（使用Graph）
            updateGraphParams(gpuGraph, item.data);
            cudaGraphLaunch(gpuGraph, stream);
            cudaEventRecord(item.gpuDone, stream);
        }

        // 2. 检查是否有完成的GPU工作
        for (auto &item : workQueue)
        {
            if (item.currentStage == GPU_PROCESS)
            {
                if (cudaEventQuery(item.gpuDone) == cudaSuccess)
                {
                    item.currentStage = CPU_POST;
                    cpuPostProcess(item.data); // 处理结果
                    // 可以回收item
                }
            }
        }

        // 3. 添加新工作项
        if (workQueue.size() < 3)
        { // 保持流水线深度
            workQueue.push(createNewWorkItem());
        }
    }
};