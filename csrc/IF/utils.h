#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor IF_forward_cu(torch::Tensor input, torch::Tensor alpha, int step, float& SOP);

std::vector<torch::Tensor> IF_backward_cu(torch::Tensor grad);

torch::Tensor IF_forward_kernel_cu(torch::Tensor input, torch::Tensor alpha, int step, float& SOP);
