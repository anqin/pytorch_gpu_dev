#include "alex_net.h"

namespace tofu {
namespace model {

AlexNetImpl::AlexNetImpl(int input_size, int input_channel) 
    : Module("AlexNet") {
    conv1 = register_module("conv1", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channel, 64, 3)
                              .padding((34 - input_size) / 2)));
    conv2 = register_module("conv2", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 3)
                              .padding(1)));
    conv3 = register_module("conv3", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3)
                              .padding(1)));
    conv4 = register_module("conv4", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3)
                              .padding(1)));
    conv5 = register_module("conv5", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3)
                              .padding(1)));

    fc1 = register_module("fc1", torch::nn::Linear(256 * 6 * 6, 4096));
    fc2 = register_module("fc2", torch::nn::Linear(4096, 4096));
    fc3 = register_module("fc3", torch::nn::Linear(4096, 10));
}

torch::Tensor AlexNetImpl::forward(torch::Tensor x) {
    namespace F = torch::nn::functional;
    x = F::max_pool2d(F::relu(conv1(x)), F::MaxPool2dFuncOptions(3).stride({2, 2}));
    x = F::max_pool2d(F::relu(conv2(x)), F::MaxPool2dFuncOptions(3).stride({2, 2}));
    x = F::relu(conv3(x));
    x = F::relu(conv4(x));
    x = F::max_pool2d(F::relu(conv5(x)), F::MaxPool2dFuncOptions(2).stride({1, 1}));
    x = x.flatten(1);
    x = F::dropout(x, F::DropoutFuncOptions().p(0.5));
    x = F::relu(fc1(x));
    x = F::dropout(x, F::DropoutFuncOptions().p(0.5));
    x = F::relu(fc2(x));
    x = F::dropout(x, F::DropoutFuncOptions().p(0.5));
    x = fc3(x);
    return x;
}


} // namespace model
} // namespace tofu
