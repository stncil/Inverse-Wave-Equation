#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <complex>
#include <cmath>

template <typename scalar_t>
__device__ __forceinline__ scalar_t complex_mul_real(scalar_t a, scalar_t b) {
    return a * b;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t complex_mul_imag(scalar_t a, scalar_t b) {
    return a * b;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t complex_exp_real(scalar_t real, scalar_t imag) {
    return exp(real) * cos(imag);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t complex_exp_imag(scalar_t real, scalar_t imag) {
    return exp(real) * sin(imag);
}

template <typename scalar_t>
__global__ void dct_blur_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ frequencies,
    const scalar_t* __restrict__ t,
    const int N) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load input data
    const scalar_t x_val = x[idx];
    const scalar_t freq_val = frequencies[idx];
    const scalar_t t_val = t[idx];
    const scalar_t zero = 0.0;

    // Compute complex exponential
    const scalar_t exp_real = complex_exp_real(zero, -freq_val * t_val);
    const scalar_t exp_imag = complex_exp_imag(zero, -freq_val * t_val);
    
    // Apply DCT blur
    const scalar_t result_real = x_val * exp_real;
    const scalar_t result_imag = x_val * exp_imag;
    
    // Store result
    output[idx] = result_real;
    output[idx + N] = result_imag;
}

torch::Tensor dct_blur_cuda(
    torch::Tensor x,
    torch::Tensor frequencies,
    torch::Tensor t) {
    
    // Ensure input tensors are float32
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(frequencies.dtype() == torch::kFloat32, "Frequencies tensor must be float32");
    TORCH_CHECK(t.dtype() == torch::kFloat32, "t tensor must be float32");
    
    const auto N = x.numel();
    auto output = torch::empty({x.size(0), 2, x.size(2), x.size(3)}, x.options());
    
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    dct_blur_kernel<float><<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        frequencies.data_ptr<float>(),
        t.data_ptr<float>(),
        N
    );
    
    return output;
}

torch::Tensor dct_blur_cpu(
    torch::Tensor x,
    torch::Tensor frequencies,
    torch::Tensor t) {
    
    // Ensure input tensors are float32
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(frequencies.dtype() == torch::kFloat32, "Frequencies tensor must be float32");
    TORCH_CHECK(t.dtype() == torch::kFloat32, "t tensor must be float32");
    
    const auto N = x.numel();
    auto output = torch::empty({x.size(0), 2, x.size(2), x.size(3)}, x.options());
    
    // Get data pointers
    auto x_data = x.data_ptr<float>();
    auto freq_data = frequencies.data_ptr<float>();
    auto t_data = t.data_ptr<float>();
    auto output_data = output.data_ptr<float>();
    
    // CPU implementation
    for (int i = 0; i < N; ++i) {
        const float x_val = x_data[i];
        const float freq_val = freq_data[i];
        const float t_val = t_data[i];
        
        const float exp_real = std::cos(-freq_val * t_val);
        const float exp_imag = std::sin(-freq_val * t_val);
        
        output_data[i] = x_val * exp_real;
        output_data[i + N] = x_val * exp_imag;
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dct_blur_cuda", &dct_blur_cuda, "DCT blur (CUDA)");
    m.def("dct_blur_cpu", &dct_blur_cpu, "DCT blur (CPU)");
} 