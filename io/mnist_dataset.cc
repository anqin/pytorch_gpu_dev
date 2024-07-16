/* ====================================================
#   Copyright (C) 2024 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : mnist_dataset.cc
#   Last Modified : 2024-07-16 16:53
#   Describe      : 
#
# ====================================================*/

#include "io/mnist_dataset.h"

#include <torch/torch.h>
#include <glog/logging.h>

#include "io/path.h"
#include "executor/simple_executor.h"

namespace tofu {
namespace io {



MnistDataSet::MnistDataSet(const std::string& dataset_path,
                           executor::SimpleExecutor* executor)
    : dataset_path_(dataset_path), executor_(executor) {}

bool MnistDataSet::IsPathExist() {
    return !(IsExist(dataset_path_) && IsDir(dataset_path_));
}

bool MnistDataSet::Download() { return false; }

bool MnistDataSet::Train(uint32_t batch_size) {
    if (!IsPathExist()) {
        LOG(ERROR) << "Invalid path: " << dataset_path_;
        return false;
    }

    std::string train_dataset_file = dataset_path_ + "/train-images-idx3-ubyte";
    auto train_data_set = torch::data::datasets::MNIST(train_dataset_file, 
                                                       torch::data::datasets::MNIST::Mode::kTrain)
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());
    const std::size_t train_dataset_size = train_data_set.size().value();
    LOG(INFO) << "MINST dataset loaded,"
        << train_dataset_size
        << " training samples found.";
    // constexpr double learning_rate = 1e-2;
    auto train_loader = torch::data::make_data_loader(std::move(train_data_set), batch_size);

  
    // auto loss = std::make_unique<torch::nn::CrossEntropyLoss>(new torch::nn::CrossEntropyLoss());
    executor::TrainContext train_context;
    train_context.learning_rate_ = 1e-2;
    executor_->InitTrainContext(&train_context);
    // train_context.loss_ = std::move(std::make_shared<torch::nn::CrossEntropyLoss>(new torch::nn::CrossEntropyLoss()));
    // train_context.loss_ = std::move(loss);
    // train_context.optimizer_ = new torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(learning_rate).momentum(0.9));
    return false;
}

bool MnistDataSet::Predict(uint32_t batch_size) {
    return false;
}

} // namespace io
} // namespace tofu
