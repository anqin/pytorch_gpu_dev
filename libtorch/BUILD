cc_library(
    name = "torch_all",
    hdrs = ["torch/torch.h"],
    export_incs=[
        "include",
        "include/torch/csrc/api/include"
    ],
    deps = [
        ':torch',
        ':torch_cpu',
        ':torch_global_deps',
        ':c10'
    ],
    warning = 'no',
    visibility = ['PUBLIC']
)

cc_library(
    name = 'c10',
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
    name = 'torch_global_deps',
    hdrs = [],
    prebuilt=True
)
