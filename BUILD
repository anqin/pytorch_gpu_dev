cc_library(
    name = 'torch_lib',
    hdrs = ['torch/torch.h'],
    visibility = ['PUBLIC'],
    export_incs=[
        "include",
        "include/torch/csrc/api/include"
    ],
    deps = [
        '//libtorch:torch',
        '//libtorch:torch_cpu',
        '//libtorch:torch_cuda',
        '//libtorch:c10',
        '//libtorch:c10_cuda',
        '//libtorch:kineto',
        '#cuda',
        '#nvrtc',
        '#nvToolsExt',
        '#cudart'
    ],
    extra_cppflags = ['-D_GLIBCXX_USE_CXX11_ABI=0'],
    extra_linkflags = ['-L/lib/intel64', '-L/usr/local/cuda/lib64'],
    warning = 'no'

)

cc_binary(
    name = 'mnist_main',
    srcs = 'mnist.cpp',
    deps = [
        '//libtorch:torch',
        '//libtorch:torch_cpu',
        '//libtorch:torch_cuda',
        '//libtorch:c10',
        '//libtorch:c10_cuda',
        '//libtorch:kineto',
        '#cuda',
        '#nvrtc',
        '#nvToolsExt',
        '#cudart'
    ],
    extra_cppflags = ['-D_GLIBCXX_USE_CXX11_ABI=0'],
    extra_linkflags = ['-L/lib/intel64', '-L/usr/local/cuda/lib64'],
    warning = 'no'
)


cc_binary(
    name = 'mnist_simple_main',
    srcs = 'mnist_simple.cpp',
    deps = [
        '//libtorch:torch',
        '//libtorch:torch_cpu',
        '//libtorch:torch_cuda',
        '//libtorch:c10',
        '//libtorch:c10_cuda',
        '//libtorch:kineto',
        '#cuda',
        '#nvrtc',
        '#nvToolsExt',
        '#cudart'
    ],
    extra_cppflags = ['-D_GLIBCXX_USE_CXX11_ABI=0'],
    extra_linkflags = ['-L/lib/intel64', '-L/usr/local/cuda/lib64'],
    warning = 'no'
)


cc_binary(
    name = 'load_model_main',
    srcs = 'load_model.cpp',
    deps = [
        '//libtorch:torch',
        '//libtorch:torch_cpu',
        '//libtorch:torch_cuda',
        '//libtorch:c10',
        '//libtorch:c10_cuda',
        '//libtorch:kineto',
        '#cuda',
        '#nvrtc',
        '#nvToolsExt',
        '#cudart'
    ],
    extra_cppflags = ['-D_GLIBCXX_USE_CXX11_ABI=0'],
    extra_linkflags = ['-L/lib/intel64', '-L/usr/local/cuda/lib64'],
    warning = 'no'
)


cc_binary(
    name = 'gpu_test',
    srcs = 'gpu_test.cpp',
    deps = [
        '//libtorch:torch',
        '//libtorch:torch_cpu',
        '//libtorch:torch_cuda',
        '//libtorch:c10',
        '//libtorch:c10_cuda',
        '//libtorch:kineto',
        '#cuda',
        '#nvrtc',
        '#nvToolsExt',
        '#cudart'
    ],
    extra_cppflags = ['-D_GLIBCXX_USE_CXX11_ABI=0'],
    extra_linkflags = ['-L/lib/intel64', '-L/usr/local/cuda/lib64'],
    warning = 'no'
)

cc_library(
    name = 'tofu_flags',
    hdrs = [],
    srcs = 'tofu_flags.cc',
    visibility = ['PUBLIC']
)
