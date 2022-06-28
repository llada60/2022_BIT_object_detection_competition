from mmdet.datasets.coco import CocoDataset as _CocoDataset


class CocoDataset(_CocoDataset):
    def __init__(self, config, classes, data_root, test_mode=False):
        ann_file = config['ann_file']
        pipeline = config['pipeline']
        img_prefix = config['img_prefix']
        super(CocoDataset, self).__init__(ann_file=ann_file,
                                          pipeline=pipeline,
                                          classes=classes,
                                          data_root=data_root,
                                          img_prefix=img_prefix,
                                          test_mode=test_mode)

    def result2csv(self, results):
        csv_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = bboxes[i][:4]
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label] - 1
                    csv_results.append(data)
        return csv_results
