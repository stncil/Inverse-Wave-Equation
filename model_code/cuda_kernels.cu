#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

template <typename scalar_t>
__global__ void wave_equation_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ kx,
    const scalar_t* __restrict__ ky,
    const scalar_t* __restrict__ t,
    const scalar_t mass,
    const scalar_t c,
    const scalar_t lamda,
    const scalar_t gamma,
    const int batch_size,
    const int height,
    const int width) {
    
    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h >= height || w >= width) return;
    
    // Optimized memory access pattern
    const int spatial_idx = h * width + w;
    const int idx = b * height * width + spatial_idx;
    
    // Load data with coalesced access
    const scalar_t x_val = x[idx];
    const scalar_t kx_val = kx[spatial_idx];
    const scalar_t ky_val = ky[spatial_idx];
    const scalar_t t_val = t[b];
    
    // Precompute common terms
    const scalar_t k_squared = kx_val * kx_val + ky_val * ky_val;
    const scalar_t k_sqrt = sqrt(k_squared);
    
    // Compute dispersion and dissipation terms
    const scalar_t dispersion = sqrt(mass * mass + c * c * k_squared + lamda * k_squared * k_squared);
    const scalar_t dissipation = gamma * k_sqrt;
    
    // Compute exponential term using fast math
    const scalar_t exp_term = __expf(-dispersion * t_val) * __cosf(dissipation * t_val);
    
    // Apply wave equation
    output[idx] = x_val * exp_term;
}

torch::Tensor wave_equation_cuda(
    torch::Tensor x,
    torch::Tensor kx,
    torch::Tensor ky,
    torch::Tensor t,
    double mass,
    double c,
    double lamda,
    double gamma,
    int block_x = 32,
    int block_y = 8) {
    
    // Ensure input tensors are float32
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(kx.dtype() == torch::kFloat32, "kx tensor must be float32");
    TORCH_CHECK(ky.dtype() == torch::kFloat32, "ky tensor must be float32");
    TORCH_CHECK(t.dtype() == torch::kFloat32, "t tensor must be float32");
    
    const auto batch_size = x.size(0);
    const auto height = x.size(2);
    const auto width = x.size(3);
    
    auto output = torch::empty_like(x);
    
    // Configurable thread block configuration
    dim3 threads(block_x, block_y);
    dim3 blocks((width + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y,
                batch_size);
    
    wave_equation_kernel<float><<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        kx.data_ptr<float>(),
        ky.data_ptr<float>(),
        t.data_ptr<float>(),
        static_cast<float>(mass),
        static_cast<float>(c),
        static_cast<float>(lamda),
        static_cast<float>(gamma),
        batch_size,
        height,
        width
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wave_equation_cuda", &wave_equation_cuda, "Wave equation (CUDA)",
          py::arg("x"),
          py::arg("kx"),
          py::arg("ky"),
          py::arg("t"),
          py::arg("mass") = 0.0,
          py::arg("c") = 1.0,
          py::arg("lamda") = 0.5,
          py::arg("gamma") = 0.1,
          py::arg("block_x") = 32,
          py::arg("block_y") = 8);
} 