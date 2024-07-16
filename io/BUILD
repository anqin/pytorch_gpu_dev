cc_library(
    name = 'mnist_dataset',
    hdrs = 'mnist_dataset.h',
    srcs = 'mnist_dataset.cc',
    deps = [
        ':path',
        '#gflags',
        '#glog',
        '//libtorch:torch',
        '//libtorch:torch_cpu',
        '//libtorch:torch_cuda',
    ],
    warning = 'no'
)

cc_library(
    name = 'path',
    hdrs = 'path.h',
    srcs = 'path.cc',
    warning = 'no'
)
