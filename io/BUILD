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
        '//executor:executor',
        '//libtorch:c10',
        '//libtorch:c10_cuda',
        '//libtorch:kineto',
        '//:tofu_flags',
        '//executor:executor'
    ],
    visibility = ['PUBLIC'],
    warning = 'no'
)

cc_library(
    name = 'path',
    hdrs = 'path.h',
    srcs = 'path.cc',
    visibility = ['PUBLIC'],
    warning = 'no'
)
