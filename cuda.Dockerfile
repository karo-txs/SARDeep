FROM cnstark/pytorch:1.13.0-py3.9.12-cuda11.7.1-ubuntu20.04

WORKDIR /SARDeep
COPY . /SARDeep

RUN cd /SARDeep

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=America/Recife apt-get -y install tzdata
RUN apt-get install ffmpeg libsm6 libxext6 -y && \
    apt-get clean && \
    apt-get install git -y &&  \
    apt-get install vim -y

# Install requirements
RUN pip install pycocotools-fix && \
    pip install -r requirements_complete.txt

# Install MMEngine and MMCV
RUN pip install openmim && \
    pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# Install MMDetection
RUN cd src/ && \
    git clone -b 2.x --single-branch https://github.com/open-mmlab/mmdetection.git && \
    cd mmdetection/ && \
    pip install -e .

ENV CHECKPOINTS_PATH=src/infra/checkpoints
ENV RESOURCE_PATH=resource
ENV CONFIG_PATH=src/infra/configs
ENV WORK_DIR=../results
ENV DATA_ROOT=src/mmdetection/data
ENV DEVICE=cuda
ENV DATA_PATH=../datasets
ENV BASE_DIR=src

WORKDIR /SARDeep