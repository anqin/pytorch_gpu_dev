cc_library(
    name = "torch_all",
    hdrs = ["torch/torch.h"],
    export_incs=[
        "include",
        "include/torch/csrc/api/include"
    ],
    deps = [
        ':c10',
        ':c10_cuda',
        ':torch_cpu',
        ':torch_cuda',
        ':torch_global_deps',
        ':torch',
        ':caffe2_nvrtc',
        '#cuda'
    ],
    warning = 'no',
    link_all_symbols=True,
    defs = ['-Wl,--no-as-needed'],
    visibility = ['PUBLIC']
)

cc_library(
    name = 'c10',
    hdrs = [],
    visibility = ['PUBLIC'],
#    link_all_symbols=True,
    prebuilt=True
)

cc_library(
    name = 'c10_cuda',
    hdrs = [],
#    link_all_symbols=True,
    prebuilt=True
)

cc_library(
    name = 'c10d_cuda_test',
    hdrs = [],
    prebuilt=True
)

cc_library(
    name = 'torch',
    hdrs = [],
    prebuilt=True
)

cc_library(
    name = 'torch_cpu',
    hdrs = [],
    prebuilt=True
)

cc_library(
    name = 'torch_cuda',
    hdrs = [],
    prebuilt=True
)

cc_library(
    name = 'torch_cuda_linalg',
    hdrs = [],
    prebuilt=True
)

cc_library(
    name = 'torch_global_deps',
    hdrs = [],
    prebuilt=True
)

cc_library(
    name = 'caffe2_nvrtc',
    hdrs = [],
    prebuilt=True
)


