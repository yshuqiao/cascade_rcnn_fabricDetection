import time, os
import json
import mmcv
from mmdet.apis import init_detector, inference_detector
import numpy as np



def main():
    #config_file = 'configs/cascade_mask_rcnn_r101_fpn_1x.py'  # config file
    #config_file = 'configs/cascade_rcnn_r50_fpn_1x.py'  # 修改成自己的配置文件
    config_file = 'configs/cascade_rcnn_r101_fpn_1x.py'  # 修改成自己的配置文件

    #checkpoint_file = 'work_dirs/cascade_mask_rcnn_r101_fpn_1x/epoch_12.pth'  # checkpoint file
    #checkpoint_file = "work_dirs/cascade_rcnn_r50_fpn_1x/epoch_12.pth"
    #checkpoint_file = "work_dirs/cascade_rcnn_r50_fpn_1x_Msize/epoch_12.pth"
    #checkpoint_file = "work_dirs/cascade_mask_rcnn_r101_fpn_1x_addcoco/epoch_12.pth"
    checkpoint_file = "work_dirs/cascade_rcnn_r101_fpn_1x/epoch_12.pth"

    #test_path = 'data/coco/test'  # 官方测试集图片路径
    test_path = '/root/guangdong1_round1_testA_20190818'  # data path

    json_name = "result_" + "" + time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".json"

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_list = []
    for img_name in os.listdir(test_path):
        if img_name.endswith('.jpg'):
            img_list.append(img_name)

    result = []
    from tqdm import tqdm
    for i, img_name in tqdm(enumerate(img_list, 1)):
        full_img = os.path.join(test_path, img_name)
        predict = inference_detector(model, full_img)
        #predict = predict[0]   #necessary for cascade mask rcnn
        for i, bboxes in enumerate(predict, 1):
            if len(bboxes) > 0:
                defect_label = i
                image_name = img_name
                for bbox in bboxes:
                    #if len(bbox) > 0:
                    if float(bbox[4]) > 0:
                        #print(bbox)

                        #[x1, y1, x2, y2, score] = bbox  #for cascade mask rcnn
                        x1, y1, x2, y2, score = bbox.tolist()  #for cascade rcnn

                        x1, y1, x2, y2 = round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)  # save 0.00
                        result.append({'name': image_name, 'category': int(defect_label), 'bbox': [x1, y1, x2, y2], 'score': float(score)})

    with open(json_name, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    main()