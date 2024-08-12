#include <iostream>
#include <string>
#include <filesystem>


#include "simple_net.h"
// #include "lenet5.h"
// #include "alex_net.h"

int main(int argc, const char *argv[])
{
    std::string mnist_dataset_path = "./data";
    std::string mnist_cls_model_path = "SimpleNet.pt";

    // try
    // {
    //     boost::program_options::options_description test_options_desc("Model testing options");
    //     test_options_desc.add_options()("help,h", "help guide")("path,p", boost::program_options::value<std::string>(&mnist_dataset_path)->required(), "path to MNIST dataset")("model,m", boost::program_options::value<std::string>(&mnist_cls_model_path)->required(), "path to MNIST Classification model");
    //     boost::program_options::variables_map vm;
    //
    //     if (argc < 2)
    //     {
    //         std::cerr << test_options_desc << std::endl;
    //         return -1;
    //     }
    //     boost::program_options::store(boost::program_options::parse_command_line(argc, argv, test_options_desc), vm);
    //
    //     if (vm.count("help") > 0)
    //     {
    //         std::cout << test_options_desc << std::endl;
    //         return -1;
    //     }
    //     boost::program_options::notify(vm);
    // }
    // catch (const std::exception &e)
    // {
    //     std::cout << e.what() << std::endl;
    //     return -1;
    // }

    // if (std::filesystem::exists(std::filesystem::path(mnist_dataset_path).append("t10k-images-idx3-ubyte")) == false)
    // {
    //     std::cout << "MNIST dataset path check failed! check path!" << std::endl;
    //     std::cout << "ERROR: " << mnist_dataset_path << "/t10k-images-idx3-ubyte does not exist!" << std::endl;
    //     return -1;
    // }
    // if (std::filesystem::exists(mnist_cls_model_path) == false)
    // {
    //     std::cout << "MNIST model check failed! check path!" << std::endl;
    //     std::cout << "ERROR: " << mnist_cls_model_path << " does not exist!" << std::endl;
    //     return -1;
    // }
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
    {
        std::cout << torch::cuda::device_count() << " cuda is available, use GPU." << std::endl;
        device = torch::kCUDA;
    }
    else
    {
        std::cout << "cuda is NOT available, use CPU." << std::endl;
    }

    auto test_data_set = torch::data::datasets::MNIST(mnist_dataset_path, torch::data::datasets::MNIST::Mode::kTest).map(torch::data::transforms::Normalize<>(0.5, 0.5));
    const std::size_t test_dataset_size = test_data_set.size().value();
    std::cout << "MINST dataset loaded, " << test_dataset_size << " testing samples found." << std::endl;

    constexpr std::size_t batch_size = 64;
    auto test_loader = torch::data::make_data_loader(std::move(test_data_set), batch_size);

    SimpleNet model(28 * 28, 300, 100, 10);
    // LeNet5 model(28);
    // AlexNet model(28, 1);

    model->to(device);

    torch::serialize::InputArchive ar;
    ar.load_from(mnist_cls_model_path);
    model->load(ar);
    std::cout << "start testing" << std::endl;
    model->eval();
    int test_correct = 0;
    for (auto &batch : *test_loader)
    {
        for (auto &sample : batch)
        {
            auto inputs = sample.data.to(device);
            auto labels = sample.target.to(device);
            inputs = inputs.unsqueeze(0); // [1, 1, 28, 28]
            auto outputs = model(inputs);
            auto [value, id] = torch::max(outputs.data(), 1);
            test_correct += torch::sum(id == labels).item().toInt();
        }
    }
    std::cout << "correct: " << 100.0f * test_correct / test_dataset_size << std::endl;
    return 0;
}
