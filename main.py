import os
import random
import tqdm
from pycocotools.coco import COCO
import cv2
import numpy as np

'''
├─annotations
└─images
  ├─train2017
  └─val2017
'''

def true_or_false():
    return bool(random.randint(0, 1))

def main():
    # random.seed('0101')
    dataset_path = '/Users/zhangchao/datasets/MSCOCO2017'
    img_path = os.path.join(dataset_path, 'val2017')
    manipulated_path = os.path.join(dataset_path, 'manipulated')
    mask_path = os.path.join(dataset_path, 'manipulated_masks')
    if not os.path.exists(manipulated_path):
        os.makedirs(manipulated_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    coco = COCO(os.path.join(dataset_path, 'annotations', 'instances_val2017.json'))
    imgIds = coco.getImgIds()
    print('Total images: ', len(imgIds))

    # for imgId in tqdm.tqdm(imgIds, ncols=150):
    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        file_name = img['file_name']
        h = img['height']  # int
        w = img['width']
        # id = img['id']
        print('object: ', file_name,h,w)  # 000000397133.jpg 427 640
        
        img_data = cv2.imread(os.path.join(img_path, file_name), cv2.IMREAD_UNCHANGED)
        # cv2.imshow('image', img_data)
        
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        if len(annIds) > 0:
            sel_idx = random.randint(0, len(annIds)-1)
            anns = coco.loadAnns(annIds)
            mask = coco.annToMask(anns[2]) * 255  # TODO del
            # mask = coco.annToMask(anns[sel_idx]) * 255
            # print(mask.shape)  # h, w
            # print(mask.dtype)  # uint8
            y,x = mask[:,:].nonzero() # get the nonzero alpha coordinates
            minx = np.min(x)
            miny = np.min(y)
            maxx = np.max(x)
            maxy = np.max(y)
            croped_mask = mask[miny:maxy, minx:maxx]
            if croped_mask.shape[0]*croped_mask.shape[1]<15:
                # TODO choose another instance
                continue
            
            background_id = imgId if true_or_false() else random.choice(imgIds)
            print('object id: ', imgId, 'background id: ', background_id)
            background = coco.loadImgs(background_id)[0]
            background_h = background['height']
            background_w = background['width']
            print('background info: ', background_h, background_w)

            mask_h = maxy-miny
            mask_w = maxx-minx
            if background_h < mask_h or background_w < mask_w:
                # TODO choose another background
                continue
            # print(mask_h, mask_w)

            new_mask = np.zeros(shape=(background_h, background_w),dtype=np.uint8)
            
            print(new_mask.shape)
            # new_y = 10
            new_y = random.randint(0, background_h - mask_h)
            # new_x = 10
            new_x = random.randint(0, background_w - mask_w)
            new_mask[new_y:new_y+mask_h,new_x:new_x+mask_w] = croped_mask
            print('new: ',new_y, new_x)

            # cv2.imshow('image', img_data)
            # cv2.imshow('d', mask)
            cv2.imshow('d1', new_mask)
            cv2.imshow('d2', mask)
            cv2.waitKey(0)
        break


if __name__ == '__main__':
    main()
