## 1. Setup Environment
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

cd ..
git clone https://github.com/open-mmlab/mmyolo.git

cd mmyolo
pip install -r requirements/albu.txt
mim install -v -e .
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

5. Run setup.sh
````commandline
sh setup.sh
````