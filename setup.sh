#!/bin/bash
datasets=sard
checkpoints=retinanet_r18_fpn_1x_coco,centernet_resnet18_140e_coco,fovea_r50_fpn_4x4_1x_coco,ddod_r50_fpn_1x_coco
labels=person

echo "1/5 Create labels.txt"
if [ ! -d "datasets/labels.txt" ]; then
  for i in ${labels//,/ }
    do
      printf "$i\n" >> "datasets/labels.txt"
    done
fi

echo "2/5 Convert format datasets to MSCOCO"
for i in ${datasets//,/ }
  do
    if [ ! -d "datasets/$i/coco" ]; then

      echo "2/5 Create paths"
      mkdir "datasets/$i/coco"
      mkdir "datasets/$i/coco/annotations"
      mkdir "datasets/$i/coco/images/"
      mkdir "datasets/$i/coco/images/train2017"
      mkdir "datasets/$i/coco/images/val2017"
      mkdir "datasets/$i/coco/images/test2017"

      echo "2/5 Convert train dataset to coco format"
      python src/infra/scripts/voc_to_coco.py \
      --ann_dir "datasets/$i/VOC2012/Annotations" \
      --ann_ids "datasets/$i/VOC2012/ImageSets/Main/train.txt" \
      --labels "datasets/labels.txt" \
      --output "datasets/$i/coco/annotations/instances_train2017.json"

      echo "2/5 Convert val dataset to coco format"
      python src/infra/scripts/voc_to_coco.py \
      --ann_dir "datasets/$i/VOC2012/Annotations" \
      --ann_ids "datasets/$i/VOC2012/ImageSets/Main/val.txt" \
      --labels "datasets/labels.txt" \
      --output "datasets/$i/coco/annotations/instances_val2017.json"

      echo "2/5 Convert test dataset to coco format"
      python src/infra/scripts/voc_to_coco.py \
      --ann_dir "datasets/$i/VOC2012/Annotations" \
      --ann_ids "datasets/$i/VOC2012/ImageSets/Main/test.txt" \
      --labels "datasets/labels.txt" \
      --output "datasets/$i/coco/annotations/instances_test2017.json"

      echo "2/5 Copy train images"
      python src/infra/scripts/copy_images.py \
      --base "datasets/$i" \
      --dst "train2017" \
      --file "train.txt"

      echo "2/5 Copy val images"
      python src/infra/scripts/copy_images.py \
      --base "datasets/$i" \
      --dst "val2017" \
      --file "val.txt"

      echo "2/5 Copy test images"
      python src/infra/scripts/copy_images.py \
      --base "datasets/$i" \
      --dst "test2017" \
      --file "test.txt"
    fi
  done

echo "2/4 Copy datasets to mmdetection path"
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

echo "3/4 Download checkpoints"

for i in ${checkpoints//,/ }
  do
    if [ ! -d "checkpoints/$i" ]; then
      echo "Downloading: $i"
      python src/infra/scripts/download_weights.py --weights $i
    fi
  done

echo "4/4 Classes configuration"
python src/infra/scripts/config_classes.py



