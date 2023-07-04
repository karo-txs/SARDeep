ARG PYTORCH="1.13.0"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# Install the required packages
RUN apt-get update \
    && apt-get install ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && -rf /var/lib/apt/lists/*

# Install MMEngine and MMCV
RUN pip install requirements.txt && \
    pip install openmim && \
    pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# Install MMDetection
RUN cd /src && \
    git clone -b 2.x --single-branch https://github.com/open-mmlab/mmdetection.git && \
    cd /mmdetection && \
    pip install -e .

ENV CHECKPOINTS_PATH=../infra/checkpoints
ENV RESOURCE_PATH=../../resource
ENV CONFIG_PATH=../../src/infra/configs
ENV WORK_DIR=../../results
ENV DATA_ROOT=../mmdetection/data
ENV DEVICE=cuda
ENV DATA_PATH=../../datasets
ENV BASE_DIR=/SARDeep/src

RUN python /content/SARDeep/src/infra/scripts/config_classes.py