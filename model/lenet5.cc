#include "lenet5.h"

namespace tofu {
namespace model {

LeNet5Impl::LeNet5Impl(int input_size) 
    : Module("LeNet5") {
    C1 = register_module("C1", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 6, 5)
                              .padding((32 - input_size) / 2)));
    C3 = register_module("C3", torch::nn::Conv2d(6, 16, 5));
    C5 = register_module("C5", torch::nn::Conv2d(16, 120, 5));
    F6 = register_module("F6", torch::nn::Linear(120, 84));
    OUTPUT = register_module("OUTPUT", torch::nn::Linear(84, 10));
}

torch::Tensor LeNet5Impl::forward(torch::Tensor x) {
    namespace F = torch::nn::functional;
    x = F::max_pool2d(F::relu(C1(x)), F::MaxPool2dFuncOptions({2, 2})); // C1 S2
    x = F::max_pool2d(F::relu(C3(x)), F::MaxPool2dFuncOptions({2, 2})); // C3 S4
    x = F::relu(C5(x));                                                 // C5
    x = F::relu(F6(x.flatten(1)));                                      // F6
    x = OUTPUT(x);                                                      // OUTPUT
    return x;
}


} // namespace model
} // namespace tofu
