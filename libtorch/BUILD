# NOTE: the follow session shoul be pasted in the BUILD file,
#        where the binary (execuable file)
#
#cc_binary(
#    name = '<binary name>',
#    srcs = '<binary srcs>',
#    deps = [
#        '//libtorch:torch',
#        '//libtorch:torch_cpu',
#        '//libtorch:torch_cuda',
#        '//libtorch:c10',
#        '//libtorch:c10_cuda',
#        '//libtorch:kineto',
#        '#cuda',
#        '#nvrtc',
#        '#nvToolsExt',
#        '#cudart'
#    ],
#    extra_cppflags = ['-D_GLIBCXX_USE_CXX11_ABI=0'],
#    extra_linkflags = ['-L/lib/intel64', '-L/usr/local/cuda/lib64'],
#    warning = 'no'
#)

cc_library(
    name = 'torch',
    hdrs = ['torch/torch.h'],
    visibility = ['PUBLIC'],
    export_incs=[
        "include",
        "include/torch/csrc/api/include"
    ],
    extra_linkflags = ['-Wl,--as-needed'],
    prebuilt=True
)

cc_library(
    name = 'torch_cpu',
    hdrs = [],
    visibility = ['PUBLIC'],
    extra_linkflags = ['-Wl,--as-needed'],
    prebuilt=True
)

cc_library(
    name = 'torch_cuda',
    hdrs = [],
    visibility = ['PUBLIC'],
    extra_linkflags = ['-Wl,--as-needed'],
    prebuilt=True
)

cc_library(
    name = 'c10',
    hdrs = [],
    visibility = ['PUBLIC'],
    prebuilt=True
)

cc_library(
    name = 'c10_cuda',
    hdrs = [],
    visibility = ['PUBLIC'],
    prebuilt=True
)

cc_library(
    name = 'kineto',
    hdrs = [],
    visibility = ['PUBLIC'],
    prebuilt=True
)
