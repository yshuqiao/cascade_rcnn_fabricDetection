import os, json
from PIL import Image, ImageDraw
from tqdm import tqdm
# base_dir = '../../../data/guangdong/' # path to your data dir
##base_dir = 'root/coco/'  # path to your data dir

def vflip(dataset, path, name):
    save_dir = path + 'defect_Images_vflip'
    os.makedirs(save_dir, exist_ok=True)
    for img_info in tqdm(dataset['images']):
        img = Image.open(os.path.join(path, 'defect_Images', img_info['file_name']))
        img = img.transpose(1)
        img.save(os.path.join(save_dir, img_info['file_name']))

    image_id2wh = {i['id']: [i['width'], i['height']] for i in dataset['images']}

    for anno_info in tqdm(dataset['annotations']):
        w, h = image_id2wh[anno_info['image_id']]
        anno_info['bbox'][1] = h - anno_info['bbox'][1] - anno_info['bbox'][3]

        for idx, seg in enumerate(anno_info['segmentation'][0]):
            if idx % 2 == 1:
                anno_info['segmentation'][0][idx] = h - seg

    json.dump(dataset, open(path + '{}_vflip.json'.format(name),'w'))


def rotate180(dataset, path, name):
    save_dir = path + 'defect_Images_rotate180'
    os.makedirs(save_dir, exist_ok=True)
    for img_info in tqdm(dataset['images']):
        img = Image.open(os.path.join(path, 'defect_Images', img_info['file_name']))
        img = img.transpose(3)
        img.save(os.path.join(save_dir, img_info['file_name']))

    image_id2wh = {i['id']: [i['width'], i['height']] for i in dataset['images']}

    for anno_info in tqdm(dataset['annotations']):
        w,h = image_id2wh[anno_info['image_id']]
        anno_info['bbox'] = [w - anno_info['bbox'][0] - anno_info['bbox'][2], 
                             h - anno_info['bbox'][1] - anno_info['bbox'][3] , 
                             anno_info['bbox'][2], 
                             anno_info['bbox'][3]]
        
        #for idx, seg in enumerate(anno_info['segmentation'][0]):
        for idx, seg in enumerate(anno_info['segmentation']):
            if idx % 2 == 1:
                anno_info['segmentation'][0][idx-1], anno_info['segmentation'][0][idx] = \
                w- anno_info['segmentation'][0][idx-1], 
                h - anno_info['segmentation'][0][idx]

    json.dump(dataset, open(path + '{}_rotate180.json'.format(name), 'w'))

path = '/root/coco/'
name = 'annotations/coco_train'
dataset = json.load(open(path + '{}.json'.format(name)))
vflip(dataset, path, name)
rotate180(dataset, path, name)

# path = '../../../data/guangdong/guangdong1_round1_train2_20190828/'
# name = 'train2'
# dataset = json.load(open(path + '{}.json'.format(name)))
# vflip(dataset, path, name)
# rotate180(dataset, path, name)