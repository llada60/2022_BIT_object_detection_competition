import os
import json
from config import *
from PIL import Image
from matplotlib import pyplot as plt


def show_sample(show_img_id):
    anno = open(os.path.join(data_root, 'annotations/train.json'), 'rt', encoding='UTF-8')
    anno = json.load(anno)

    categories_name = ['']
    categories_name.extend([category['name'] for category in anno['categories']])

    show_img_anno = []
    for elem in anno['annotations']:
        if elem['image_id'] == show_img_id:
            show_img_anno.append(elem)

    img = Image.open(data_root + 'train/%07d.jpg' % show_img_id)

    show_img_bbox = []
    show_img_label = []
    for elem in show_img_anno:
        show_img_label.append(categories_name[elem['category_id']])
        show_img_bbox.append(elem['bbox'])

    colors = ['b', 'g', 'r', 'm', 'c']
    text_color = 'w'

    plt.figure(figsize=(15, 15))

    fig = plt.imshow(img)
    for idx, bbox in enumerate(show_img_bbox):
        color = colors[idx % len(colors)]
        rect = plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2], height=bbox[3], fill=False,
                             edgecolor=color, linewidth=2)
        fig.axes.add_patch(rect)
        fig.axes.text(bbox[0], bbox[1], show_img_label[idx], va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
    plt.show()


if __name__ == '__main__':
    show_sample(10)
