/* ====================================================
#   Copyright (C) 2024 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : gpu_test.cpp
#   Last Modified : 2024-07-11 15:42
#   Describe      : 
#
# ====================================================*/

#include <iostream>

#include <torch/torch.h>

int main() {
    torch::Device device(torch::kCPU);
    torch::Tensor tensor = torch::zeros({2, 2});
    std::cout << tensor << std::endl;


    if (torch::cuda::is_available()) {
      std::cout << "CUDA is available! " << std::endl;
      device = torch::kCUDA;
    }

    torch::Tensor test_gpu_tensor = tensor.to(device);

    std::cout << test_gpu_tensor << std::endl;
    return 0;
}
