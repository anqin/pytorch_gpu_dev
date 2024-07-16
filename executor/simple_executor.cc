/* ====================================================
#   Copyright (C) 2024 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : simple_executor.cc
#   Last Modified : 2024-07-16 18:13
#   Describe      : 
#
# ====================================================*/

#include "executor/simple_executor.h"

#include "model/alex_net.h"

namespace tofu {
namespace executor {

SimpleExecutor::SimpleExecutor() 
    : device_(torch::kCPU), model_(new model::AlexNet(28, 1)) {
    // model_ = std::make_shared<AlexNet*>(new AlexNet(28, 1));
}

void SimpleExecutor::SetupDevice() {
    if (torch::cuda::is_available()) {
        LOG(INFO) << torch::cuda::device_count()
            << " cuda is available, use GPU.";
        device_ = torch::kCUDA;
    }
}

void SimpleExecutor::InitTrainContext(TrainContext* context) {
    context->optimizer_ = std::make_unique<torch::optim::SGD>(
                            (*model_)->parameters(), 
                            torch::optim::SGDOptions(
                                context->learning_rate_)
                            .momentum(0.9));
    context->criterion_ = std::make_unique<torch::nn::CrossEntropyLoss>();

    SetupDevice();
    (*model_)->to(device_);
    (*model_)->train();
}

void SimpleExecutor::Train(torch::Tensor& input, torch::Tensor& label, 
                           TrainContext* context) {
    auto input_tensor = input.to(device_);
    auto input_label = label.to(device_);
    context->optimizer_->zero_grad();
    auto outputs = (*model_)(input_tensor);
    // auto outputs = (*model_)->forward(inputs);
    auto loss = (*context->criterion_)(outputs, input_label);
    loss.backward();
    context->optimizer_->step();

    context->sum_loss_ += loss.item().toDouble();
    auto [value, id] = torch::max(outputs.data(), 1);
    context->train_correct_ += torch::sum(id == input_label).item().toInt();
}


void SimpleExecutor::SaveModel(const std::string& model_file) {
    torch::serialize::OutputArchive ar;
    (*model_)->save(ar);

    const std::string local_file = (*model_)->name() + ".pt";
    ar.save_to(local_file);

    LOG(INFO) << "Trained model saved to: " << local_file;
}

} // namespace executor
} // namespace tofu
