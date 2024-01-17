#include <torch/extension.h>
#include <iostream>
#include <cmath>
#include <vector>
#include "utils.h"

torch::Tensor IF_forward_cu(torch::Tensor input, torch::Tensor alpha, int step, float& SOP)
{
	CHECK_INPUT(input);
	CHECK_INPUT(alpha);
	return IF_forward_kernel_cu(input, alpha, step, SOP);
}

// We don't need the backward function for IF neurons.
std::vector<torch::Tensor> IF_backward_cu(torch::Tensor grad)
{
	CHECK_INPUT(grad);
	return {torch::zeros(grad.sizes(), grad.options()), torch::tensor(0, grad.options())};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward_cu", &IF_forward_cu);
	m.def("backward_cu", &IF_backward_cu);
}
