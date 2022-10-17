FROM python:3.9

WORKDIR /app
COPY . .

RUN pip3 install cmake
RUN pip3 install dlib
RUN pip3 install -r requirement.txt
RUN python -m pip install detectron2 -f \https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update -y
RUN apt-get install -y kmod
ARG nvidia_binary_version="515.65.01"
ARG nvidia_binary="NVIDIA-Linux-x86_64-${nvidia_binary_version}.run"
RUN wget -q https://us.download.nvidia.com/XFree86/Linux-x86_64/${nvidia_binary_version}/${nvidia_binary} && chmod +x ${nvidia_binary} && ./${nvidia_binary} --accept-license --ui=none --no-kernel-module --no-questions && rm -rf ${nvidia_binary}
CMD ["python3", "Pipeline.py"]