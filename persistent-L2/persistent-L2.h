#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, char const* const func, char const* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


#define MIN(x, y) (x <= y ? x : y)


template <typename T> 
void ping_address_to_L2(T* global_address, size_t size, cudaStream_t* stream_ptr, float hit_ratio = 1.0){
    cudaDeviceProp dprops;
    int current_device = 0;
    CHECK_CUDA_ERROR(cudaGetDevice(&current_device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&dprops, current_device));
    bool is_applied = dprops.major >= 8; // persistent L2 is applied on sm_8x and later
    if (!is_applied)
    {
        std::cerr << "persistent L2 is not applied on sm_" << dprops.major << dprops.minor << std::endl;
        return;
    }
    cudaCtxResetPersistingL2Cache();
    const size_t required_size = size * sizeof(T);
    const size_t persist_l2_limit = dprops.persistingL2CacheMaxSize;
    const size_t window_size = MIN(required_size, persist_l2_limit);

    cudaStreamAttrValue stream_attr;
    stream_attr.accessPolicyWindow.base_ptr = global_address;
    stream_attr.accessPolicyWindow.num_bytes = window_size;
    stream_attr.accessPolicyWindow.hitRatio = hit_ratio;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    CHECK_CUDA_ERROR(cudaStreamSetAttribute(*stream_ptr, cudaStreamAttributeAccessPolicyWindow, &stream_attr));
}
