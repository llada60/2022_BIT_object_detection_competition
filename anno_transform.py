import cv2
import json
from config import *


if __name__ == '__main__':
    anno_path = data_root + 'annotations/original_train.json'
    train_anno_path = data_root + 'annotations/train.json'
    test_anno_path = data_root + 'annotations/test.json'

    # train annotations
    with open(anno_path, 'r') as f:
        train_anno = json.load(f)

    ori_bboxes = train_anno['annotations']
    for anno in ori_bboxes:
        x, y, w, h = anno['bbox']
        anno['bbox'] = [x, y, w, h]
        anno['segmentation'] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        anno['area'] = w * h
        anno['category_id'] += 1
        anno['iscrowd'] = 0

    categories = train_anno['categories']
    for category in categories:
        category['id'] += 1
        category['supercategory'] = 'None'

    with open(train_anno_path, 'w') as f:
        json.dump(train_anno, f, indent=1, separators=(',', ': '))

    # test annotations
    img_list = []
    for i in range(33354, 49716):
        filename = '00' + str(i) + '.jpg'
        img = cv2.imread(data_root + 'test/' + filename)
        img_info = dict(
            file_name=filename,
            height=img.shape[0],
            width=img.shape[1],
            id=i
        )
        img_list.append(img_info)

    test_anno = dict(
        info=train_anno['info'],
        images=img_list,
        annotations=[],
        categories=train_anno['categories']
    )

    with open(test_anno_path, 'w') as f:
        json.dump(test_anno, f, indent=1, separators=(',', ':'))
