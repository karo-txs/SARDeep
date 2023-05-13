#!/bin/bash
datasets=sard
checkpoints=retinanet_r18_fpn_1x_coco,centernet_resnet18_140e_coco,fovea_r50_fpn_4x4_1x_coco,ddod_r50_fpn_1x_coco

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



