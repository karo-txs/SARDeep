import xml.etree.ElementTree as ET
import argparse
import glob
import re


def get_classes(data_path_mm) -> list:
    classes_names = []
    print(data_path, data_path_mm)
    for xml_file in glob.glob(f"{data_path}/{data_path_mm}/VOC2012/Annotations" + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)

    classes_names = list(set(classes_names))
    classes_names.sort()
    print(f"Classes: {classes_names}")
    return classes_names


def update_classes(classes_names: list):
    f_names = [f"{base_dir}/mmdetection/mmdet/datasets/voc.py",
               f"{base_dir}/mmdetection/mmdet/datasets/coco.py"]
    for f_name in f_names:
        with open(f_name) as f:
            s = f.read()
            s = re.sub('CLASSES = \(.*?\)',
                       'CLASSES = ({})'.format(", ".join(["{}".format(name) for name in classes_names])), s,
                       flags=re.S)
        with open(f_name, 'w') as f:
            f.write(s)
    f_names = [f"{base_dir}/mmdetection/mmdet/core/evaluation/class_names.py"]
    for f_name in f_names:
        with open(f_name) as f:
            s = f.read()
            s = re.sub("return \[.*?\]", 'return {}'.format(", ".join(["{}".format(name) for name in classes_names])),
                       s, flags=re.S)
        with open(f_name, 'w') as f:
            f.write(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasets', default='sard', help='dataset path name')
    parser.add_argument('-b', '--base-dir', default='src', help='base dir')
    parser.add_argument('-p', '--data-path', default='datasets', help='datasets path')
    args = vars(parser.parse_args())

    global base_dir
    base_dir = args['base_dir']

    global data_path
    data_path = args['data_path']

    classes = get_classes(args["datasets"])
    update_classes([classes])
