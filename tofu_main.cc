/* ====================================================
#   Copyright (C) 2024 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : tofu_main.cc
#   Last Modified : 2024-07-16 18:34
#   Describe      : 
#
# ====================================================*/

#include "io/mnist_dataset.h"
#include "executor/simple_executor.h"

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: ./exe <path-to-mnist-dataset>\n";
        return -1;
    }

    tofu::executor::SimpleExecutor executor;
    tofu::io::MnistDataSet mnist_trainer(argv[1], &executor);

    mnist_trainer.Train();

    return 0;
}
