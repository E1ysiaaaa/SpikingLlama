#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

template <typename scalar_t>
__global__ void IF_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> input,
    const scalar_t alpha,
	int step, float& SOP,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> output
)
{
	const int T = input.size(0), B = input.size(1), S = input.size(2), D = input.size(3);
	const int b = blockIdx.x * blockDim.x + threadIdx.x;
	const int s = blockIdx.y * blockDim.y + threadIdx.y;
	const int d = blockIdx.z * blockDim.z + threadIdx.z;

	if (b >= B || s >= S || d >= D)
		return;
	
	scalar_t threshold = alpha;
	scalar_t membrane = 0.5 * threshold;
	for (int i = 0; i < T; i += step)
	{
		for (int j = 0; j < min(step, T - i); j++)
			membrane += input[i+j][b][s][d];
		for (int j = 0; j < min(step, T - i); j++)
		{
			scalar_t spike = 0;
			if (membrane > threshold)
			{
				spike = 1;
				membrane = membrane - threshold;
				SOP += 1;
			}
			output[i+j][b][s][d] = spike * threshold;
		}
	}
}

torch::Tensor IF_forward_kernel_cu(torch::Tensor input, torch::Tensor alpha, int step, float& SOP)
{
	const int T = input.size(0), B = input.size(1), S = input.size(2), D = input.size(3);
	auto output = torch::zeros({T, B, S, D}, input.options());
	
	const dim3 threads(4, 8, 8);
	const dim3 blocks((B + threads.x - 1)/threads.x, (S + threads.y - 1)/threads.y, (D + threads.z - 1)/threads.z);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "IF_forward_kernel_cu",
    ([&] {
        IF_fw_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            alpha.item<scalar_t>(),
			step, SOP,
            output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>()
        );
    }));
	return output;
}
