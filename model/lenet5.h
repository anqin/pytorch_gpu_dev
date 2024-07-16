#pragma once

#include <torch/torch.h>

namespace tofu {
namespace model {

/// @brief Implement of LeNet5,
/// according to LeCun Y, Bottou L, Bengio Y, et al. 
///      Gradient-based learning applied to document recognition[J]. 
///      Proceedings of the IEEE, 1998, 86(11): 2278-2324.
class LeNet5Impl : public torch::nn::Module
{
public:
    /// @brief Constructor
    /// @param input_size the width of image, padding to 32 * 32
    LeNet5Impl(int input_size);

    /// @brief forward function
    /// @param x input tensor
    /// @return output tensor
    torch::Tensor forward(torch::Tensor x);

private:
    /// Pooling layer S2 and S4 used in forward step
    torch::nn::Conv2d C1{nullptr}, C3{nullptr}, C5{nullptr};
    torch::nn::Linear F6{nullptr}, OUTPUT{nullptr};
};

TORCH_MODULE(LeNet5);


} // namespace model
} // namespace tofu
