cc_library(
    name = 'executor',
    hdrs = 'simple_executor.h',
    srcs = 'simple_executor.cc',
    deps = [
        '#gflags',
        '#glog',
        '//libtorch:torch',
        '//libtorch:torch_cpu',
        '//libtorch:torch_cuda',
        '//model:model'
    ],
    visibility = ['PUBLIC'],
    warning = 'no'

)
