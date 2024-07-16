cc_library(
    name = 'model',
    hdrs = [
        'simple_net.h',
        'alex_net.h',
        'lenet5.h'
    ],
    srcs = [
        'simple_net.cc',
        'alex_net.cc',
        'lenet5.cc'
    ],
    deps = [
        '#gflags',
        '#glog',
        '//libtorch:torch',
        '//libtorch:torch_cpu',
        '//libtorch:torch_cuda',
    ],
    visibility = ['PUBLIC'],
    warning = 'no'

)
