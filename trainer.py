import csv
import time
import mmcv
import datetime
import torch.optim
from config import *
from dataset import CocoDataset
from mmdet.datasets import build_dataloader
from cascade_rcnn import CascadeRCNN


class Trainer:
    def __init__(self):
        self.train_dataset = CocoDataset(train_data_config, classes, data_root)
        self.test_dataset = CocoDataset(test_data_config, classes, data_root, test_mode=True)

        self.train_dataloader = build_dataloader(dataset=self.train_dataset, samples_per_gpu=4, workers_per_gpu=2)
        self.val_dataloader = build_dataloader(dataset=self.train_dataset, samples_per_gpu=1, workers_per_gpu=2,
                                               shuffle=False)
        self.test_dataloader = build_dataloader(dataset=self.test_dataset, samples_per_gpu=1, workers_per_gpu=2,
                                                shuffle=False)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = CascadeRCNN(model).to(self.device)
        self.optimizer = torch.optim.SGD(
            params=self.detector.parameters(),
            lr=2e-3,
            momentum=0.9,
            weight_decay=1e-4
        )
        self.epoch_num = epoch_num
        self.test_result_path = test_result_path
        self.model_path = model_path
        self.loss_log = dict(
            loss=0.0,
            cls=0.0,
            bbox=0.0,
            s0_cls=0.0,
            s0_acc=0.0,
            s0_bbox=0.0,
            s1_cls=0.0,
            s1_acc=0.0,
            s1_bbox=0.0,
            s2_cls=0.0,
            s2_acc=0.0,
            s2_bbox=0.0,
            batch_num=0
        )

    def train(self):
        print('Train batch number:', len(self.train_dataloader))
        start_time = time.time()
        for idx in range(self.epoch_num):
            print('batch   loss    cls     bbox    s0_cls  s0_acc  s0_bbox s1_cls  s1_acc  s1_bbox s2_cls  s2_acc  s2_bbox time')
            for i, data in enumerate(self.train_dataloader):
                for key in data:
                    data[key] = data[key].data
                data['img'] = data['img'][0].to(self.device)
                data['img_metas'] = data['img_metas'][0]
                data['gt_bboxes'] = data['gt_bboxes'][0]
                data['gt_labels'] = data['gt_labels'][0]
                bbox_num = len(data['gt_bboxes'])
                for j in range(bbox_num):
                    data['gt_bboxes'][j] = data['gt_bboxes'][j].to(self.device)
                label_num = len(data['gt_bboxes'])
                for j in range(label_num):
                    data['gt_labels'][j] = data['gt_labels'][j].to(self.device)

                result = self.detector.train_step(data)
                self.update_loss(result)

                result['loss'].backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if (i + 1) % 100 == 0:
                    self.show_loss(i + 1)
                    end_time = time.time()
                    print('{:<8d}'.format(round(end_time - start_time)))
                    start_time = end_time

            self.test(mode='val')

    def test(self, mode='test'):
        results = []
        dataloader = None
        if mode == 'val':
            dataloader = self.val_dataloader
            dataset = dataloader.dataset
        elif mode == 'test':
            dataloader = self.test_dataloader
            dataset = dataloader.dataset

        prog_bar = mmcv.ProgressBar(len(dataset))

        for i, data in enumerate(dataloader):
            if mode == 'val':
                for key in data:
                    data[key] = data[key].data
                    if key == 'img':
                        for j in range(len(data[key])):
                            data[key][j] = data[key][j].to(self.device)
                    if key == 'gt_bboxes' or key == 'gt_labels':
                        for j in range(len(data[key])):
                            for k in range(len(data[key][j])):
                                data[key][j][k] = data[key][j][k].to(self.device)
            else:
                for j in range(len(data['img_metas'])):
                    data['img_metas'][j] = data['img_metas'][j].data[0]
                for j in range(len(data['img'])):
                    data['img'][j] = data['img'][j].to(self.device)

            with torch.no_grad():
                result = self.detector.aug_test(data['img'], data['img_metas'], rescale=True)

            batch_size = len(result)
            results.extend(result)

            for _ in range(batch_size):
                prog_bar.update()

        if mode == 'val':
            metric = dataset.evaluate(results, classwise=True)
            for key in metric:
                if key != 'bbox_mAP_copypaste':
                    print(key, metric[key])
        else:
            results = dataset.result2csv(results)
            self.save_test_result(results)

    def save_test_result(self, prediction):
        t = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file = self.test_result_path + t + '.csv'
        with open(file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'predictions'])
            for i in range(0, 1000):
                prediction_info = ''
                for item in prediction:
                    if item['image_id'] == i:
                        bbox_info = '{' + str(item['bbox'][0]) + ' ' + str(item['bbox'][1]) + ' ' + str(
                            item['bbox'][2]) + ' ' + str(item['bbox'][3]) + ' ' + str(
                            item['score']) + ' ' + str(item['category_id']) + '}'
                        prediction_info += bbox_info
                writer.writerow([i, prediction_info])

    def update_loss(self, loss):
        self.loss_log['loss'] += loss['loss']
        log_vars = loss['log_vars']
        self.loss_log['cls'] += log_vars['loss_cls']
        self.loss_log['bbox'] += log_vars['loss_bbox']
        self.loss_log['s0_cls'] += log_vars['s0.loss_cls']
        self.loss_log['s0_acc'] += log_vars['s0.acc']
        self.loss_log['s0_bbox'] += log_vars['s0.loss_bbox']
        self.loss_log['s1_cls'] += log_vars['s1.loss_cls']
        self.loss_log['s1_acc'] += log_vars['s1.acc']
        self.loss_log['s1_bbox'] += log_vars['s1.loss_bbox']
        self.loss_log['s2_cls'] += log_vars['s2.loss_cls']
        self.loss_log['s2_acc'] += log_vars['s2.acc']
        self.loss_log['s2_bbox'] += log_vars['s2.loss_bbox']
        self.loss_log['batch_num'] += 1

    def show_loss(self, batch_no):
        factor = self.loss_log['batch_num']
        loss_info = '{:<8d}'.format(batch_no)
        for key in self.loss_log:
            if key != 'batch_num':
                info = '{:<8.3f}'.format(self.loss_log[key] / factor)
                loss_info += info
            self.loss_log[key] = 0
        print(loss_info, end='')

    def save_model(self):
        t = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file = self.model_path + t + '.pth'
        state_dict = self.detector.state_dict()
        torch.save(state_dict, file)
