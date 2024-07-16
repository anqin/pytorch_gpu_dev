/* ====================================================
#   Copyright (C) 2024 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : simple_executor.h
#   Last Modified : 2024-07-16 17:05
#   Describe      : 
#
# ====================================================*/


#pragma once

#include <memory>
#include <torch/torch.h>

namespace tofu {

namespace model {
class AlexNet;
}

namespace executor {

struct TrainContext {
    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    std::unique_ptr<torch::nn::CrossEntropyLoss> loss_;

    double sum_loss_;
    int train_correct_;
    double learning_rate_;

    TrainContext() : 
        optimizer_(nullptr), loss_(nullptr),
        sum_loss_(0.0), train_correct_(0), learning_rate_(1e-2) {}
};

class SimpleExecutor {
public:
    SimpleExecutor();
    ~SimpleExecutor() {}

    void InitTrainContext(TrainContext* context);

private:
    std::shared_ptr<model::AlexNet> model_;
    // model::AlexNet model_;
};

} // namespace executor
} // namespace tofu
