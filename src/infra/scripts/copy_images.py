import argparse
import shutil


def copy_images(base_path: str, file_name: str, dst_path: str):
    with open(f"{base_path}/VOC2012/ImageSets/Main/{file_name}") as f:
        lines = f.readlines()

    for line in lines:
        file_name = line.replace("\n", "")
        shutil.copyfile(f"{base_path}/VOC2012/JPEGImages/{file_name}.jpg", f"{base_path}/coco/images/{dst_path}/{file_name}.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base', default='datasets/sard', help='dataset path name')
    parser.add_argument('-d', '--dst', default='train2017', help='dst path')
    parser.add_argument('-f', '--file', default='train.txt', help='file name')
    args = vars(parser.parse_args())

    copy_images(args["base"], args["file"], args["dst"])
