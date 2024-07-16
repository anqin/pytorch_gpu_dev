/* ====================================================
#   Copyright (C) 2024 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : mnist_dataset.cc
#   Last Modified : 2024-07-16 18:29
#   Describe      : 
#
# ====================================================*/

#include "io/mnist_dataset.h"

#include <torch/torch.h>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "io/path.h"
#include "executor/simple_executor.h"

DECLARE_int32(tofu_executor_train_epoch_num);
DECLARE_int32(tofu_executor_train_batch_size);

namespace tofu {
namespace io {



MnistDataSet::MnistDataSet(const std::string& dataset_path,
                           executor::SimpleExecutor* executor)
    : dataset_path_(dataset_path), executor_(executor) {}

bool MnistDataSet::IsPathExist() {
    return !(IsExist(dataset_path_) && IsDir(dataset_path_));
}

bool MnistDataSet::Download() { return false; }

bool MnistDataSet::Train() {
    if (!IsPathExist()) {
        LOG(ERROR) << "Invalid path: " << dataset_path_;
        return false;
    }

    uint32_t batch_size = FLAGS_tofu_executor_train_batch_size;
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
    auto train_loader = torch::data::make_data_loader(
                                std::move(train_data_set), batch_size);

  
    executor::TrainContext train_context;
    train_context.learning_rate_ = 1e-2;
    executor_->InitTrainContext(&train_context);
    

    LOG(INFO) << "Start training with setting: ["
        << "epoch: " << FLAGS_tofu_executor_train_epoch_num
        << ", batch size: " << batch_size
        << ", learing rate: " << train_context.learning_rate_
        << "]";
    auto time_start = std::chrono::system_clock::now();
    for (int32_t i = 1; i <= FLAGS_tofu_executor_train_epoch_num; i++) {
        train_context.sum_loss_ = 0.0;
        train_context.train_correct_ = 0;

        for (auto &batch : *train_loader) {
            executor_->Train(batch.data, batch.target, &train_context);
        
        }
        LOG(INFO) << "[" << i << " / " << FLAGS_tofu_executor_train_epoch_num 
            << "] loss: " << train_context.sum_loss_ / (train_dataset_size / batch_size)
            << ", correct: " << 100.0f * train_context.train_correct_ / train_dataset_size;
    }
    auto time_end = std::chrono::system_clock::now();
    LOG(INFO) << "Train time: " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()
        << "ms";

    LOG(INFO) << "Saving trained model...";
    executor_->SaveModel();

    return true;
}

bool MnistDataSet::Predict(uint32_t batch_size) {
    return false;
}

} // namespace io
} // namespace tofu
