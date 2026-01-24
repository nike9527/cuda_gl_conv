#include <cuda_runtime.h>
#include <iostream>
#include <vector>

class CudaGraphPipeline
{
private:
    cudaStream_t stream_;
    cudaGraph_t graph_;
    cudaGraphExec_t graphExec_;
    float *d_input_;
    float *d_output_;
    size_t size_;

public:
    CudaGraphPipeline(size_t size) : size_(size)
    {
        // 1. 创建流
        cudaStreamCreate(&stream_);

        // 2. 分配持久化内存
        cudaMalloc(&d_input_, size_ * sizeof(float));
        cudaMalloc(&d_output_, size_ * sizeof(float));

        // 3. 录制Graph
        recordGraph();
    }

    ~CudaGraphPipeline()
    {
        cudaGraphExecDestroy(graphExec_);
        cudaGraphDestroy(graph_);
        cudaFree(d_input_);
        cudaFree(d_output_);
        cudaStreamDestroy(stream_);
    }

private:
    void recordGraph()
    {
        // 清理旧的Graph
        if (graphExec_)
            cudaGraphExecDestroy(graphExec_);
        if (graph_)
            cudaGraphDestroy(graph_);

        // 开始录制
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

        // 操作1: 初始化输入
        initKernel<<<(size_ + 255) / 256, 256, 0, stream_>>>(d_input_, size_);

        // 操作2: 处理数据
        processKernel<<<(size_ + 255) / 256, 256, 0, stream_>>>(d_input_, d_output_, size_);

        // 操作3: 后处理
        postProcessKernel<<<(size_ + 255) / 256, 256, 0, stream_>>>(d_output_, size_);

        // 结束录制
        cudaStreamEndCapture(stream_, &graph_);

        // 实例化
        cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0);
    }

public:
    // 执行Graph
    void execute()
    {
        cudaGraphLaunch(graphExec_, stream_);
    }

    // 等待完成（可选）
    void synchronize()
    {
        cudaStreamSynchronize(stream_);
    }

    // 更新数据（如果需要）
    void updateInput(const std::vector<float> &data)
    {
        // 使用异步拷贝更新输入
        cudaMemcpyAsync(d_input_, data.data(),
                        data.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);

        // 如果需要改变Graph结构，重新录制
        if (data.size() != size_)
        {
            size_ = data.size();
            recordGraph(); // 重新录制
        }
    }
};

// 使用示例
int main()
{
    const size_t N = 1000000;
    CudaGraphPipeline pipeline(N);

    // 预热
    for (int i = 0; i < 10; i++)
    {
        pipeline.execute();
    }
    pipeline.synchronize();

    // 性能测试
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++)
    {
        pipeline.execute(); // 极低开销
    }
    pipeline.synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Time per Graph launch: "
              << std::chrono::duration<double, std::micro>(end - start).count() / 1000
              << " us" << std::endl;

    return 0;
}