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
    return False
    return bool(random.randint(0, 1))

def main():
    random.seed('0101')
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

    for imgId in tqdm.tqdm(imgIds, ncols=150):
    # for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        file_name = img['file_name']
        h = img['height']
        w = img['width']
        img_data = cv2.imread(os.path.join(img_path, file_name), cv2.IMREAD_UNCHANGED)
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        if len(annIds) > 0:
            sel_idx = random.randint(0, len(annIds)-1)
            anns = coco.loadAnns(annIds)
            mask = coco.annToMask(anns[sel_idx]) * 255
            y,x = mask[:,:].nonzero()
            minx = np.min(x)
            miny = np.min(y)
            maxx = np.max(x)
            maxy = np.max(y)
            croped_mask = mask[miny:maxy, minx:maxx]
            if croped_mask.shape[0]*croped_mask.shape[1]<15:
                # TODO choose another instance
                continue
            
            background_id = imgId if true_or_false() else random.choice(imgIds)
            background = coco.loadImgs(background_id)[0]
            background_h = background['height']
            background_w = background['width']

            mask_h = maxy-miny
            mask_w = maxx-minx
            if background_h < mask_h or background_w < mask_w:
                # TODO choose another background
                continue

            new_mask = np.zeros(shape=(background_h, background_w),dtype=np.uint8)
            new_y = random.randint(0, background_h - mask_h)
            new_x = random.randint(0, background_w - mask_w)
            new_mask[new_y:new_y+mask_h,new_x:new_x+mask_w] = croped_mask
            new_mask = cv2.merge([new_mask, new_mask, new_mask])

            instance = img_data[miny:maxy, minx:maxx]
            # TODO soft
            # croped_mask = cv2.merge([croped_mask, croped_mask, croped_mask])
            # instance = cv2.bitwise_and(instance, croped_mask)
            background_data = cv2.imread(os.path.join(img_path, background['file_name']), cv2.IMREAD_UNCHANGED)
            # new_background = cv2.bitwise_and(background_data, cv2.bitwise_not(new_mask))
            # new_background[new_y:new_y+mask_h,new_x:new_x+mask_w] = cv2.add(new_background[new_y:new_y+mask_h,new_x:new_x+mask_w], instance)
            # TODO save img and mask
            # background_data[new_mask == 0] = 0
            # blurred_image = cv2.blur(background_data, ksize=(3,3))
            # blurred_mask = cv2.blur(new_mask,ksize=(3,3))
            # result = blurred_image / blurred_mask
            
            # cv2.imshow('d2', result)
            print('\n')
            print('background_data: ',background_data.shape)
            print('mask_h: ',mask_h, 'mask_w: ',mask_w)
            print('new_y: ',new_y, 'new_x: ',new_x)
            print('p1: ',new_y+(mask_h//2),'p2: ', new_x+(mask_w//2))
            print('instance: ', instance.shape)
            print('croped_mask: ', croped_mask.shape)


            new_background = cv2.seamlessClone(img_data, background_data, mask, (new_y+(mask_h//2), new_x+(mask_w//2)), flags=cv2.MONOCHROME_TRANSFER)
            
            # new_background = cv2.seamlessClone(src = img_data, dst = background_data, mask = mask, p=(background_data.shape[0]//2, background_data.shape[1]//2), flags=cv2.MONOCHROME_TRANSFER)
            cv2.imshow('d1', croped_mask)
            cv2.imshow('d111', new_mask)
            cv2.imshow('d1.1', instance)
            cv2.imshow('d2', background_data)
            cv2.imshow('d3', new_background)
            cv2.waitKey(0)
        # break


if __name__ == '__main__':
    main()
