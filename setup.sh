#!/bin/bash
datasets=sard,heridal
checkpoints=faster_rcnn_r50_fpn_1x_coco,yolov3_d53_fp16_mstrain-608_273e_coco

echo "1/6 Environment Activation"
python3 -m venv venv
activate () {
  . ../venv/Scripts/activate # venv/bin/activate [linux]
}

echo "2/6 Install requirements"
pip install -r requirements.txt

echo "3/6 Configure mmdetection framework"
if [ ! -d "base/mmdetection" ]; then
  mim install mmcv-full
  cd base && git clone https://github.com/open-mmlab/mmdetection.git
  cd base/mmdetection && pip install -e .
  cd base/mmdetection && python setup.py install
fi

echo "4/6 Copy datasets to mmdetection path"
for i in ${datasets//,/ }
  do
    if [ ! -d "base/mmdetection/data/" ]; then
      mkdir "base/mmdetection/data"
    fi

    if [ ! -d "base/mmdetection/data/$i" ]; then
      echo "Copying $i"
      mkdir "base/mmdetection/data/$i"
      cp -R "datasets/$i" "base/mmdetection/data/"
    fi
  done

echo "5/6 Download checkpoints"

for i in ${checkpoints//,/ }
  do
    if [ ! -d "base/checkpoints/$i" ]; then
      echo "Downloading: $i"
      python base/scripts/download_weights.py --weights $i
    fi
  done

echo "6/6 Classes configuration"
python base/scripts/config_classes.py



