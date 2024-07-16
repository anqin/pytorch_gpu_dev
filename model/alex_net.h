#pragma once

#include <torch/torch.h>


namespace tofu {
namespace model {

/// @brief Implement of AlexNet
/// according to Krizhevsky A, Sutskever I, Hinton G E. 
/// ImageNet classification with deep convolutional neural networks[C]
/// International Conference on Neural Information Processing Systems. 
/// Curran Associates Inc. 2012:1097-1105. with a little modified.
class AlexNetImpl : public torch::nn::Module {
public:
    /// @brief Constructor
    /// @param input_size the width of image, padding to 32 * 32
    /// @param input_channel the channel of input image
    AlexNetImpl(int input_size, int input_channel);

    /// @brief forward function
    /// @param x input tensor
    /// @return output tensor
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, 
        conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

TORCH_MODULE(AlexNet);


} // namespace model
} // namespace tofu
