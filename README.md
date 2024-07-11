GPU-dev
=======

1. download cuda-specific libtorch (2.3.1 with cuda 11.8)
  
   $ wget https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.3.1%2Bcu118.zip

2. download typhoon-blade, which is the latest version

   $ git clone git@github.com:anqin/blade-build.git 

3. download the GPU docker

   $ docker pull registry.cn-hangzhou.aliyuncs.com/anqindev/ubuntu22-dev:ubuntu22-tf2.15-dev1 

CPU-dev
=======

1. download cuda-specific libtorch (2.3.1 with cuda 11.8)
  
   $ wget https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.3.1%2Bcu118.zip

2. download typhoon-blade, which is the v2.0-alpha

   $ git clone -b v2.0-alpha git@github.com:anqin/blade-build.git 

3. download the GPU docker

   $ docker pull registry.cn-hangzhou.aliyuncs.com/anqindev/ubuntu22-dev:

Build
======

4. path tree would be:

   copy the BUILD.libtorh (for GPU or CPU) to corrent dir:

   pytorch_gpu_dev
      |- BLADE_ROOT
      |- BUILD
      |- *.cpp/*.h
      |- libtorch_cpu
            |- BUILD
            |- <unzip the libtorch cpu lib>
      |- libtorch_gpu
            |- BUILD
            |- <unzip the libtorch GPU lib>

5. mount the dirs to docker image

   * list the docker images and find the <image ID>

   * for GPU

     $ bash docker_env_setup_gpu.sh <image id>

   * for CPU

     $ bash docker_env_setup.sh <image id>

6. build the codes

   $ blade build ...

7. run the example codes

   $ export LD_LIBRARY_PATH=<path/to/libtorch/lib64>
   $ ./mnist_sample_main
   $ ./load_model_main net.pt
