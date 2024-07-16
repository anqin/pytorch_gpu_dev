/* ====================================================
#   Copyright (C) 2024 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : simple_executor.cc
#   Last Modified : 2024-07-16 17:05
#   Describe      : 
#
# ====================================================*/

#include "executor/simple_executor.h"

#include "model/alex_net.h"

namespace tofu {
namespace executor {

SimpleExecutor::SimpleExecutor() : model_(new model::AlexNet(28, 1)) {
    // model_ = std::make_shared<AlexNet*>(new AlexNet(28, 1));
}

void SimpleExecutor::InitTrainContext(TrainContext* context) {
    context->optimizer_ = std::make_unique<torch::optim::SGD>(
                            (*model_)->parameters(), 
                            torch::optim::SGDOptions(
                                context->learning_rate_)
                            .momentum(0.9));
    context->loss_ = std::make_unique<torch::nn::CrossEntropyLoss>();
}

} // namespace executor
} // namespace tofu
