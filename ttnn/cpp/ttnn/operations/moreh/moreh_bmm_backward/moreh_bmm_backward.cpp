// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bmm_backward.hpp"

#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn {

std::vector<std::optional<Tensor>> moreh_bmm_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mat2,
    const std::vector<bool>& are_required_outputs,
    const std::optional<Tensor>& input_grad,
    const std::optional<Tensor>& mat2_grad,
    // Unused, as in moreh_matmul: each grad is written into its required output tensor, in that tensor's memory config.
    [[maybe_unused]] const std::optional<MemoryConfig>& input_grad_memory_config,
    [[maybe_unused]] const std::optional<MemoryConfig>& mat2_grad_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    // moreh_matmul's default compute config, so a caller that passes none keeps the same numerics.
    const auto compute_kernel = init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config);
    std::vector<std::optional<Tensor>> outputs(2);
    if (are_required_outputs.at(0)) {
        TT_FATAL(input_grad.has_value(), "input_grad needs to have a value when input_requires_grad is True.");
        outputs[0] = ttnn::matmul(
            output_grad,
            mat2,
            /*transpose_a=*/false,
            /*transpose_b=*/true,
            input_grad->memory_config(),
            /*dtype=*/std::nullopt,
            /*program_config=*/std::nullopt,
            /*activation=*/std::nullopt,
            compute_kernel,
            /*core_grid=*/std::nullopt,
            /*output_tile=*/std::nullopt,
            /*optional_output_tensor=*/input_grad);
    }
    if (are_required_outputs.at(1)) {
        TT_FATAL(mat2_grad.has_value(), "mat2_grad needs to have a value when mat2_requires_grad is True.");
        outputs[1] = ttnn::matmul(
            input,
            output_grad,
            /*transpose_a=*/true,
            /*transpose_b=*/false,
            mat2_grad->memory_config(),
            /*dtype=*/std::nullopt,
            /*program_config=*/std::nullopt,
            /*activation=*/std::nullopt,
            compute_kernel,
            /*core_grid=*/std::nullopt,
            /*output_tile=*/std::nullopt,
            /*optional_output_tensor=*/mat2_grad);
    }
    return outputs;
}

}  // namespace ttnn
