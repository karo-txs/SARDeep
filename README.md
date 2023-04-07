# SARDeep
Detecting Humans in Aerial Images for Search and Rescue Operations with Deep Learning Algorithms

## Setup
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
4. Run setup.sh
````commandline
sh setup.sh
````