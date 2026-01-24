// 双缓冲数据流模式
class StreamingProcessor
{
    // 双缓冲区：一个给CPU准备，一个给GPU处理
    struct Buffer
    {
        void *hostPtr;
        void *devicePtr;
        cudaEvent_t ready;
        bool inUse = false;
    };

    Buffer buffers[2];
    cudaGraphExec_t processGraph;
    cudaStream_t stream;

public:
    void processFrames()
    {
        int current = 0;

        while (hasMoreFrames())
        {
            // 缓冲区A：CPU准备数据
            Buffer &cpuBuffer = buffers[current];
            prepareFrameData(cpuBuffer.hostPtr);

            // 异步拷贝到GPU（使用当前图）
            updateGraphWithNewData(processGraph, cpuBuffer.devicePtr);
            cudaGraphLaunch(processGraph, stream);
            cudaEventRecord(cpuBuffer.ready, stream);

            // 切换到另一个缓冲区
            current = 1 - current;

            // 检查上一个缓冲区是否完成
            Buffer &prevBuffer = buffers[current];
            if (prevBuffer.inUse)
            {
                cudaEventSynchronize(prevBuffer.ready); // 等待
                processResults(prevBuffer.hostPtr);     // CPU处理结果
                prevBuffer.inUse = false;
            }
        }
    }
};