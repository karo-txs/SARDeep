# SARDeep
Detecting Humans in Aerial Images for Search and Rescue Operations with Deep Learning Algorithms

#### Run notebook with pipeline
https://github.com/AFKaro/SARDeep/blob/main/pipeline.ipynb

#### Run with Docker
1. Upload yours datasets and checkpoints:
````commandline
Datasets in -> SARDeep/datasets
Checkpoits in -> SARDeep/src/infra/checkpoints
````

- CPU Version
````commandline
docker build -f cpu.Dockerfile -t pytorch/cpu .
````

- CUDA Version
````commandline
docker build -f cuda.Dockerfile -t pytorch/cuda .
````

- Configure
````commandline
python src/infra/scripts/copy_datasets.py
python src/infra/scripts/config_classes.py
````

#### Run Locally
1. Environment Activation
````commandline
python3 -m venv venv
./venv/Scripts/activate # source ./venv/bin/activate
````
2. Install requirements
````commandline
pip install -r requirements.txt
````
3. Configure mmdetection framework
````commandline
pip install openmim
mim install mmcv-full

cd src
git clone -b 2.x --single-branch https://github.com/open-mmlab/mmdetection.git

cd mmdetection
!pip install -e .
python setup.py install
````
4. Insert our datasets in SARDeep/datasets
````commandline
 .
 ├── datasets
 │   ├── sard
 │   │   ├── VOC2012
 │   │   │   ├── Annotations
 │   │   │   │   ├── gss6.xml
 │   │   │   │   ...
 │   │   │   ├── ImageSets
 │   │   │   │   └── Main
 │   │   │   │       ├── test.txt
 │   │   │   │       ├── train.txt
 │   │   │   │       └── val.txt
 │   │   │   └── JPEGImages
 │   │   │   │   ├── gss6.jpg
 │   │   │   │   ...
 │   │   ├── coco # if you have the coco version
 │   │   │   ├── annotations
 │   │   │   │   ├── instances_train2017.json
 │   │   │   │   ├── instances_val2017.json
 │   │   │   │   └── instances_test2017.json
 │   │   │   ├── images
 │   │   │   │   ├── train2017
 │   │   │   │   │   ├── gss6.jpg
 │   │   │   │   │    ...
 │   │   │   │   ├── val2017
 │   │   │   │   │   ├── gss6.jpg
 │   │   │   │   │    ...
 │   │   │   │   └── test2017
 │   │   │   │       ├── gss6.jpg
 │   │   │   │        ...
 .
````

5. Update env variables
````commandline
SARDeep/src/.env
````

6. Run setup.sh
````commandline
sh setup.sh
````

7. Download checkpoints in mmdetection github and upload in SARDeep/src/infra/checkpoints

### 2. Update run confs
````commandline
!python src/update_configs.py \
    --model-name faster_rcnn \
    --train-data sard \
    --test-data sard \
    --activate true \
    --fold 1 \
    --iteration 1 \
    --max-epochs 1 \
    --lr 1e-2 \
    --momentum 0.9 \
    --interval 1
````

### 3. Train Models
````commandline
!python src/main.py --step Train
````

### 4. Quantize models
````commandline
!python src/main.py --step Quantization
````

### 5. Test Models
````commandline
!python src/main.py --step Test
````
