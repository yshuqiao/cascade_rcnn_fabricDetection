##My use for mmdetection##

1.Repost(git clone https://github.com/open-mmlab/mmdetection.git)
2.Mostly,follow the original instruction(mmdetection-master/docs),especially INSTALL.md,GETTING_STARTED.ma,MODEL_ZOO.md.
3.Modified for 2019 fabric defect detection competition(https://tianchi.aliyun.com/competition/entrance/231748/introduction):
(1)use the convertTococo.py to transfer fabric data provided by the competition to coco form.
(2)use the combineJsons.py to combine json files of two ronuds of fabric data.
(3)use the flipAndRotate.py to do data augmentation.
(4)put the fabric data into /root/coco(include json files and images),then run"cd mmdetection-master
mkdir data
ln -s /root/coco/ data"
(5)revise mmdet/dataset/coco.py(note the original coco CLASSES):
CLASSES = ('1', '2', '3', '4', '5', '6',
               '7', '8', '9', '10', '11',
               '12', '13', '14', '15', '16', '17',
               '18', '19', '20')
(6，not necessary)revise mmdet/core/evaluation/class_names.py like this:
def coco_classes():
    return [
        '1', '2', '3', '4', '5', '6',
        '7', '8', '9', '10', '11',
        '12', '13', '14', '15', '16', '17',
        '18', '19', '20'
    ]
(7)revise configs/cascade_rcnn_r101_fpn_1x.py:
[change] data_root if needed
[change] train = dict(type=dataset_type,ann_file=data_root + 'annotations/instances_train2017.json',
img_prefix=data_root + 'train2017/',pipeline=train_pipeline), 
[to] train=dict(type=dataset_type,ann_file=data_root + 'annotations/coco_train.json',
img_prefix=data_root + 'defect_Images/',pipeline=train_pipeline)
[change] val = dict(type=dataset_type,ann_file=data_root + 'annotations/instances_val2017.json',img_prefix=data_root + 'val2017/',pipeline=test_pipeline) if needed
[change] num_classes=81 [to] num_classes=21
[change] img_scale=(1333, 800) [to] img_scale=(1223, 500)
[change] imgs_per_gpu=2 [to] imgs_per_gpu=1 if needed
[change] lr=0.02 [to] lr=0.0025 (lr=0.00125*batch_size=0.00125*gpu_num*imgs_per_gpu)
[change] load_from=None [to] load_from = "./checkpoints/cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth"
Then under the ubuntu path of "mmdetection-master",excute "python tools/train.py configs/cascade_rcnn_r101_fpn_1x.py".
(8)use demo2_inference.py to predict,remember to changed the path of "config_file""checkpoint_file""test_path".
