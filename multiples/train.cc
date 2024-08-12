// https://github.com/jolibrain/deepdetect/blob/724fbe330607b0d286949b49f5dee779d4ca259d/src/backends/torch/torchlib.cc


#include <memory>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>
#include <future>
#include <thread>

#include "simple_net.h"
// #include "lenet5.h"
// #include "alex_net.h"


typedef torch::data::Example<std::vector<at::Tensor>,
                           std::vector<at::Tensor>>  TorchBatch;

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

struct ModelRunner {
    uint32_t device_id_;
    double learning_rate_;
    double sum_loss_;
    int train_correct_;

    std::shared_ptr<SimpleNetImpl> module_;
    std::shared_ptr<torch::optim::Optimizer> loss_;
    std::shared_ptr<torch::nn::CrossEntropyLoss> criterion_;

    ModelRunner(uint32_t id) 
        : device_id_(id), learning_rate_(1e-2), sum_loss_(0), train_correct_(0) {
        module_ = std::make_shared<SimpleNetImpl>(28 * 28, 300, 100, 10);
        loss_ = std::make_shared<torch::optim::SGD>(
                        module_->parameters(),
                        torch::optim::SGDOptions(learning_rate_)
                        .momentum(0.9));
        criterion_ = std::make_shared<torch::nn::CrossEntropyLoss>();

    }
};


void PrepareModelRunner(const std::vector<torch::Device>& devices,
                      std::vector<ModelRunner>* runners) {
    for (size_t i = 0; i < devices.size(); ++i) {
        runners->emplace_back(ModelRunner(i));
        auto &r = runners->back();
        r.module_->to(devices[i]);
        r.module_->train();
    }
}

void ConcurrentTask(ModelRunner* runner, torch::Tensor& input,
                      torch::Tensor& labels) {
    // runner->loss_->zero_grad();
    auto outputs = runner->module_->forward(input);
    auto loss = (*runner->criterion_)(outputs, labels);

    runner->sum_loss_ += loss.item().toDouble();
    auto [value, id] = torch::max(outputs.data(), 1);
    runner->train_correct_ += torch::sum(id == labels).item().toInt();
}

int main(int argc, const char *argv[]) {
    std::string mnist_dataset_path = "./data";
    std::size_t epoch_num = 5;
    std::size_t batch_size = 32;

    std::vector<torch::Device> devices;
    GetLocalDevices(&devices);
    uint32_t main_device_id = 0;

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

    // SimpleNet model(28 * 28, 300, 100, 10);
    // LeNet5 model(28);
    // AlexNet model(28, 1);

    // model->to(devices[0]);
    // auto criterion = torch::nn::CrossEntropyLoss();
    //
    // auto optimizer = torch::optim::SGD(model->parameters(), 
    //                                    torch::optim::SGDOptions(learning_rate).momentum(0.9));


    std::vector<ModelRunner> slave_models;
    PrepareModelRunner(devices, &slave_models);

    // model->train();
    std::cout << "start training with setting: [epoch: " << epoch_num 
        << ", batch size: " << batch_size 
        << ", learing rate: " << learning_rate
        << ", device num: " << devices.size()
        << "]" << std::endl;
    auto time_start = std::chrono::system_clock::now();

    bool is_epoch = false;
    uint32_t epoch_no = 0;
    auto batch_it = train_loader->begin();
    while (epoch_no < epoch_num) {
        double sum_loss = 0.0;
        int train_correct = 0;

        std::vector<std::thread> threads; 
        for (uint32_t i = 0; i < devices.size(); ++i) {
            auto inputs = batch_it->data.to(devices[i]);
            auto labels = batch_it->target.to(devices[i]);

            // threads.push_back(std::thread([&slave_models, i, &devices, &batch_it]() {
            threads.push_back(std::thread([&slave_models, i, inputs, labels]() {
                // auto inputs = batch_it->data.to(devices[i]);
                // auto labels = batch_it->target.to(devices[i]);
                
                slave_models[i].loss_->zero_grad();
                auto outputs = slave_models[i].module_->forward(inputs);
                auto loss = (*slave_models[i].criterion_)(outputs, labels);
                loss.backward();
                slave_models[i].loss_->step();

                slave_models[i].sum_loss_ += loss.item().toDouble();
                auto [value, id] = torch::max(outputs.data(), 1);
                slave_models[i].train_correct_ += torch::sum(id == labels).item().toInt();
            }));

            ++batch_it;
            if (batch_it == train_loader->end()) {
                ++epoch_no;
                is_epoch = true;
                batch_it = train_loader->begin();

                // print statics
                for (auto& r : slave_models) {
                    sum_loss += r.sum_loss_;
                    train_correct += r.train_correct_;
                }

                std::cout << "[" << epoch_no << " / " << epoch_num << "]"
                  << " loss: " << sum_loss / (train_dataset_size / batch_size)
                  << ", correct: " << 100.0f * train_correct / train_dataset_size 
                  << std::endl;
                
                if (epoch_no == epoch_num) break;
            }
        }

        for (auto& t : threads) {
            t.join();
        }
        // std::cout << "threads create finish" << std::endl;
   
        if (!is_epoch) continue;

        std::cout << "update gradient, epoch: " << epoch_no << std::endl;
        // Reduce gradients on device #0 for each epoch
        auto params = slave_models[main_device_id].module_->parameters();
        for (uint32_t id = 0; id < devices.size(); ++id) {
            if (id == main_device_id) {
                continue;
            }
            auto params_i = slave_models[id].module_->parameters();
            for (uint32_t pi = 0; pi < params.size(); ++pi) {
                auto& grad = params[pi].mutable_grad();
                auto grad_i = params_i[pi].grad();
                grad.add_(grad_i.to(devices[main_device_id]));
            }
        }
        std::cout << "broadcast weight to all, epoch: " << epoch_no << std::endl;
        for (uint32_t id = 0; id < devices.size(); ++id) {
            if (id == main_device_id) {
                continue;
            }
            auto params_i = slave_models[id].module_->parameters();
            for (uint32_t pi = 0; pi < params.size(); ++pi) {
                torch::NoGradGuard guard;
                torch::Tensor weight = params[pi];
                params_i[pi].copy_(weight.to(devices.at(id)));
                params_i[pi].grad().zero_();
            }
        }
        is_epoch = false;
    }
/*
    for (int i = 1; i <= epoch_num; i++) {
        double sum_loss = 0.0;
        int train_correct = 0;
        for (auto &batch : *train_loader) {
            auto inputs = batch.data.to(devices[0]);
            auto labels = batch.target.to(devices[0]);
            optimizer.zero_grad();
            auto outputs = model(inputs);
            auto loss = criterion(outputs, labels);
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

*/
    auto time_end = std::chrono::system_clock::now();
    std::cout << "train time: " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() 
        << "ms" << std::endl; // 9s

    std::cout << "saving trained model..." << std::endl;
    torch::serialize::OutputArchive ar;
    slave_models[0].module_->save(ar);
    const std::string model_file = slave_models[0].module_->name() + ".pt";
    ar.save_to(model_file);
    std::cout << "trained model saved to " << model_file << std::endl;
    return 0;
}
