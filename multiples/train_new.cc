/* ====================================================
#   Copyright (C) 2024 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : train_new.cc
#   Last Modified : 2024-08-19 11:32
#   Describe      : 
#
# ====================================================*/

#include <iostream>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "simple_net.h"
// #include "lenet5.h"
// #include "alex_net.h"

#include <torch/nn/parallel/data_parallel.h>


void GetLocalDevices(std::vector<torch::Device>* devices) {
    if (!torch::cuda::is_available()) {
        std::cout << "cuda is NOT available, use CPU." << std::endl;
        devices->push_back( torch::Device(torch::DeviceType::CPU) );
        return;
    }

    for (int i = 0; i < torch::cuda::device_count(); ++i) {
        devices->push_back(torch::Device(torch::DeviceType::CUDA, i));
    }

    std::cout << torch::cuda::device_count() 
        << " cuda is available, use GPU." << std::endl;
}

int main(int argc, const char *argv[]) {
    std::string mnist_dataset_path = "./data";
    std::size_t epoch_num = 5;
    std::size_t batch_size = 32;

    std::vector<torch::Device> devices;
    GetLocalDevices(&devices);

    // torch::Device device(torch::kCPU);
    // if (torch::cuda::is_available()) {
    //     std::cout << torch::cuda::device_count() << " cuda is available, use GPU." << std::endl;
    //     device = torch::kCUDA;
    // } else {
    //     std::cout << "cuda is NOT available, use CPU." << std::endl;
    // }
    auto train_data_set = torch::data::datasets::MNIST(mnist_dataset_path, 
                                                       torch::data::datasets::MNIST::Mode::kTrain)
                              .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                              .map(torch::data::transforms::Stack<>());

    const std::size_t train_dataset_size = train_data_set.size().value();

    std::cout << "MINST dataset loaded, " << train_dataset_size 
        << " training samples found." << std::endl;

    constexpr double learning_rate = 1e-2;

    auto train_loader = torch::data::make_data_loader(std::move(train_data_set), batch_size);

    SimpleNet model(28 * 28, 300, 100, 10);
    // LeNet5 model(28);
    // AlexNet model(28, 1);

    model->to(devices[0]);
    auto criterion = torch::nn::CrossEntropyLoss();

    auto optimizer = torch::optim::SGD(model->parameters(), 
                                       torch::optim::SGDOptions(learning_rate).momentum(0.9));

    model->train();
    std::cout << "start training with setting: [epoch: " << epoch_num 
        << ", batch size: " << batch_size 
        << ", learing rate: " << learning_rate 
        << "]" << std::endl;
    auto time_start = std::chrono::system_clock::now();
    for (int i = 1; i <= epoch_num; i++) {
        double sum_loss = 0.0;
        int train_correct = 0;
        for (auto &batch : *train_loader) {
            auto inputs = batch.data.to(devices[0]);
            auto labels = batch.target.to(devices[0]);
            // optimizer.zero_grad();
            // auto outputs = model(inputs);
            // auto loss = criterion(outputs, labels);

            auto outputs = torch::nn::parallel::data_parallel(model, inputs);
            auto loss = torch::mse_loss(outputs, torch::zeros_like(outputs));
            loss.backward();
            optimizer.step();

            sum_loss += loss.item().toDouble();
            auto [value, id] = torch::max(outputs.data(), 1);
            train_correct += torch::sum(id == labels).item().toInt();
        }
        std::cout << "[" << i << " / " << epoch_num << "]"
                  << " loss: " << sum_loss / (train_dataset_size / batch_size)
                  << ", correct: " << 100.0f * train_correct / train_dataset_size << std::endl;
    }
    auto time_end = std::chrono::system_clock::now();
    std::cout << "train time: " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() 
        << "ms" << std::endl; // 9s

    std::cout << "saving trained model..." << std::endl;
    torch::serialize::OutputArchive ar;
    model->save(ar);
    const std::string model_file = model->name() + ".pt";
    ar.save_to(model_file);
    std::cout << "trained model saved to " << model_file << std::endl;
    return 0;
}
