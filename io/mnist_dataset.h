/* ====================================================
#   Copyright (C) 2024 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : mnist_dataset.h
#   Last Modified : 2024-07-16 18:29
#   Describe      : 
#
# ====================================================*/

#pragma once

#include <memory>
#include <string>

namespace tofu {

namespace executor {
class SimpleExecutor;
} 

namespace io {

class MnistDataSet {
public:
    MnistDataSet(const std::string& dataset_path,
                 executor::SimpleExecutor* executor);
    ~MnistDataSet() {}

    bool IsPathExist();
    bool Download();

    bool Train();
    bool Predict(uint32_t batch_size);

private:
    std::string dataset_path_;
    executor::SimpleExecutor* executor_;
};


} // namespace io
} // namespace tofu
