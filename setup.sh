#!/bin/bash
datasets=sard,heridal
checkpoints=faster_rcnn_r50_fpn_1x_coco,yolov3_d53_fp16_mstrain-608_273e_coco

echo "1/3 Copy datasets to mmdetection path"
for i in ${datasets//,/ }
  do
    if [ ! -d "src/mmdetection/data/" ]; then
      mkdir "src/mmdetection/data"
    fi

    if [ ! -d "src/mmdetection/data/$i" ]; then
      echo "Copying $i"
      mkdir "src/mmdetection/data/$i"
      cp -R "datasets/$i" "src/mmdetection/data/"
    fi
  done

echo "2/3 Download checkpoints"

for i in ${checkpoints//,/ }
  do
    if [ ! -d "checkpoints/$i" ]; then
      echo "Downloading: $i"
      python src/infra/scripts/download_weights.py --weights $i
    fi
  done

echo "3/3 Classes configuration"
python src/infra/scripts/config_classes.py



