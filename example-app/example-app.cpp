#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//  torch::Tensor tensor1 = torch::randn({3, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 1).requires_grad(true));
  torch::Tensor tensor1 = torch::rand({3, 4}); 
  tensor1.to(device);
  std::cout << "1: " << std::endl << tensor1 << std::endl;
  torch::Tensor tensor2 = torch::rand({3, 4});
  std::cout << "2: " << std::endl << tensor2 << std::endl;

  std::cout << "=======================" << std::endl;
  std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
  }
  return 0;
}

