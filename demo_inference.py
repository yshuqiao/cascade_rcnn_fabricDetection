import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.models import build_detector
import glob, os, json
import numpy as np
#import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #set the gpu
# 构建网络，载入模型
config_file = 'configs/cascade_rcnn_r50_fpn_1x.py' #config file
checkpoint_file = 'work_dirs/cascade_rcnn_r50_fpn_1x/epoch_12.pth' # checkpoint file

# build the model from a config file and a checkpoint file
print('loading model...')
model = init_detector(config_file, checkpoint_file, device='cuda:0')
print('loading complete!')

# 测试多张图片
path = '/root/guangdong1_round1_testA_20190818' # data path "/root/guangdong1_round1_testA_20190818/"
imgs = glob.glob(path + '/*.jpg')

result=[]
thres = 0

# for i, img in enumerate(imgs):
#     im = cv2.imread(img)
#     res = inference_detector(model,im)
for i, res in enumerate(inference_detector(model, imgs)):
    print(i, imgs[i])
    bboxes = np.vstack(res)
    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(res)]
    labels = np.concatenate(labels)
    if len(bboxes) > 0:
        for j, bbox in enumerate(bboxes):
            if float(bbox[4]) > thres:
                res_line = {'name': os.path.basename(imgs[i]), 'category': int(labels[j] + 1), 'bbox':[round(float(x),2) for x in bbox[:4]], 'score':float(bbox[4])}
                result.append(res_line)
# 写入结果
with open('result.json', 'w') as fp:
     json.dump(result, fp, indent=4, separators=(',', ': '))
print('over!')