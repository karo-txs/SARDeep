FROM cnstark/pytorch:1.13.0-py3.9.12-ubuntu20.04

WORKDIR /usr/src/SARDeep
COPY . .

RUN apt-get update && apt-get install git -y

# Install requirements
RUN pip install pycocotools-fix && pip install -r requirements.txt

# Install MMEngine and MMCV
RUN pip install openmim && \
    pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# Install MMDetection
RUN cd src/ && \
    git clone -b 2.x --single-branch https://github.com/open-mmlab/mmdetection.git && \
    cd mmdetection/ && \
    pip install -e .

ENV CHECKPOINTS_PATH=../infra/checkpoints
ENV RESOURCE_PATH=../../resource
ENV CONFIG_PATH=../../src/infra/configs
ENV WORK_DIR=../../results
ENV DATA_ROOT=../mmdetection/data
ENV DEVICE=cuda
ENV DATA_PATH=../../datasets
ENV BASE_DIR=/src

WORKDIR /SARDeep