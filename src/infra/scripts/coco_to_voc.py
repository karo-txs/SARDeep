# Author: https://github.com/KapilM26/coco2VOC/blob/master/coco2voc.py
from pycocotools.coco import COCO
from pascal_voc_writer import Writer
import argparse
import os


def coco2voc(ann_file, output_dir):
    coco = COCO(ann_file)
    cats = coco.loadCats(coco.getCatIds())
    cat_idx = {}
    for c in cats:
        cat_idx[c['id']] = c['name']
    for img in coco.imgs:
        cat_ids = coco.getCatIds()
        ann_ids = coco.getAnnIds(imgIds=[img], catIds=cat_ids)
        if len(ann_ids) > 0:
            img_f_name = coco.imgs[img]['file_name']
            image_f_name_ls = img_f_name.split('.')
            image_f_name_ls[-1] = 'xml'
            label_fname = '.'.join(image_f_name_ls)
            writer = Writer(img_f_name, coco.imgs[img]['width'], coco.imgs[img]['height'])
            anns = coco.loadAnns(ann_ids)
            for a in anns:
                bbox = a['bbox']
                bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                bbox = [str(b) for b in bbox]
                cat_name = cat_idx[a['category_id']]
                writer.addObject(cat_name, bbox[0], bbox[1], bbox[2], bbox[3])
                writer.save(output_dir + '/' + label_fname)


parser = argparse.ArgumentParser(description='Convert COCO annotations to PASCAL VOC XML annotations')
parser.add_argument('--ann_file', help='Path to annotations file')
parser.add_argument('--output_dir', help='Path to output directory where annotations are to be stored')
args = parser.parse_args()
try:
    os.mkdir(args.output_dir)
except FileExistsError:
    pass

coco2voc(ann_file=args.ann_file, output_dir=args.output_dir)
