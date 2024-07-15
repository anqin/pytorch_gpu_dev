/* ====================================================
#   Copyright (C) 2024 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : load_model.cpp
#   Last Modified : 2024-07-15 13:08
#   Describe      : 
#
# ====================================================*/


#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <thread>

#include "mnist_simple.h"

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  // torch::jit::script::Module module;
  Net net;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    // module = torch::jit::load(argv[1]);
    torch::load(net, argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "load model ok" << std::endl;

  if (torch::cuda::is_available()) {
      std::cout << "Moving model to GPU" << std::endl;
      net->to(at::kCUDA);
  } else {
      std::cout << "run in CPU" << std::endl;
  }
  auto data_loader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("./data").map(
          torch::data::transforms::Stack<>()),
      /*batch_size=*/64);

  for (auto& batch : *data_loader) {
      // at::Tensor result = net.forward({batch.data}).toTensor();
      auto t = batch.data.to(at::kCUDA);
      at::Tensor result = net->forward(t);
      std::cout << "result = " << result.slice(1, 0, 5) << std::endl;
      // auto maxResult = result.max(1);
      // auto maxIndex = std::get<1>(maxResult).item<float>();
      // auto maxOut = std::get<0>(maxResult).item<float>();
      // std::cout << "Predicted: " << maxIndex << " | " << maxOut << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1)); 
  } 
  return 0;
}
