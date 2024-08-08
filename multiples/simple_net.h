#pragma once

#include <torch/torch.h>

/// @brief a simple classifier with 3 FC layer for mnist dataset
class SimpleNetImpl : public torch::nn::Module
{
public:
    /// @brief Constructor
    /// @param in_dim the dimension of input
    /// @param n_hidden_1 the dimension of hidden layer 1
    /// @param n_hidden_2 the dimension of hidden layer 2
    /// @param out_dim the dimension of output
    SimpleNetImpl(int in_dim, int n_hidden_1, int n_hidden_2, int out_dim);

    /// @brief forward function
    /// @param x input tensor
    /// @return output tensor
    torch::Tensor forward(torch::Tensor x);

    // std::string name() const { return "SimpleNet"; }

private:
    torch::nn::Linear layer1{nullptr}, layer2{nullptr}, layer3{nullptr};
};

TORCH_MODULE(SimpleNet);
