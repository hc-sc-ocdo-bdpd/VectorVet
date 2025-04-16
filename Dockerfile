ARG CUDA_IMAGE="12.2.2-devel-ubuntu22.04"  
FROM nvidia/cuda:${CUDA_IMAGE}  
  
ENV HOST=0.0.0.0  
  
RUN apt-get update && apt-get upgrade -y \  
    && apt-get install -y --no-install-recommends \  
       git build-essential \  
       python3 python3-pip python3-dev \  
       gcc g++ gfortran wget \  
       ocl-icd-opencl-dev opencl-headers clinfo \  
       libclblast-dev libopenblas-dev \  
    && mkdir -p /etc/OpenCL/vendors \  
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd  
  
COPY . .  
  
ENV CUDA_DOCKER_ARCH=all  
ENV GGML_CUDA=1  
  
# Upgrade pip, setuptools, wheel first  
RUN python3 -m pip install --upgrade pip setuptools wheel  
  
# Explicitly install cython, numpy, scipy first
RUN python3 -m pip install Cython numpy scipy  
  
# Install general dependencies  
RUN python3 -m pip install pytest cmake scikit-build \  
    fastapi uvicorn sse-starlette pydantic-settings starlette-context  
  
# Build llama-cpp-python with CUDA support  
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python  
  
# Finally install your Python requirements  
RUN pip install -r requirements.txt  
  
CMD ["python3", "-m", "llama_cpp.server"]  