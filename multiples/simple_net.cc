#include "simple_net.h"

SimpleNetImpl::SimpleNetImpl(int in_dim, int n_hidden_1, int n_hidden_2, int out_dim) : Module("SimpleNet")
{
    layer1 = register_module("layer1", torch::nn::Linear(torch::nn::LinearOptions(in_dim, n_hidden_1)));
    layer2 = register_module("layer2", torch::nn::Linear(torch::nn::LinearOptions(n_hidden_1, n_hidden_2)));
    layer3 = register_module("layer3", torch::nn::Linear(torch::nn::LinearOptions(n_hidden_2, out_dim)));
}

torch::Tensor SimpleNetImpl::forward(torch::Tensor x)
{
    x = x.flatten(1);
    x = layer1(x);
    x = layer2(x);
    x = layer3(x);
    return x;
}