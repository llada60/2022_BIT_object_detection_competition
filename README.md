# WORK4 COCO数据集物体检测

mmdetection扒皮的结果，套用cascade_rcnn_r50_fpn的结构，resnet和fpn是手动实现的，其余直接搬运。
代码可以跑起来，也不会出bug，正向传播、反向传播计算loss都正常，但就是得不到结果/准确率为0。
目前没有找到问题所在。

anno_transform.py可以把训练集标注文件转化成coco格式，同时生成测试集标注；
config.py包含大部分模型配置，注意调整data_root属性；
train.py是程序主入口。

项目仍然依赖于mmcv和mmdet中的部分功能，故运行前需要安装mmcv-full和mmdet。


文件夹结构：
```commandline
├──Detection(本项目文件夹)
├──Prediction(存储测试集预测结果的csv文件)
├──Model(存储训练好的模型参数)
```


数据文件夹结构：
```commandline
├──annotations
│   ├── original_train.json(助教提供的标注文件)
│   ├── train.json
│   ├── test.json
├── train(存放训练集图片)
├── test(存放测试集图片)
```