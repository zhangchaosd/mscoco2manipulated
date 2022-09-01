import os
import random
import tqdm
from pycocotools.coco import COCO
import cv2
import numpy as np

'''
├─annotations
├─train2017
└─val2017
'''

def main(subset='val2017'):
    random.seed('0101')
    dataset_path = '/Users/zhangchao/datasets/MSCOCO2017'
    img_path = os.path.join(dataset_path, subset)
    manipulated_path = os.path.join(dataset_path, subset + '_manipulated')
    mask_path = os.path.join(dataset_path, subset + '_manipulated_masks')
    if not os.path.exists(manipulated_path):
        os.makedirs(manipulated_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    coco = COCO(os.path.join(dataset_path, 'annotations', 'instances_val2017.json'))
    imgIds = coco.getImgIds()
    print('Total images: ', len(imgIds))

    for imgId in tqdm.tqdm(imgIds, ncols=150):
        img = coco.loadImgs(imgId)[0]
        file_name = img['file_name']
        img_data = cv2.imread(os.path.join(img_path, file_name), cv2.IMREAD_COLOR)
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        if len(annIds) > 0:
            sel_idx = random.randint(0, len(annIds) - 1)
            anns = coco.loadAnns(annIds)
            mask = coco.annToMask(anns[sel_idx]) * 255
            y, x = mask[:, :].nonzero()
            minx = np.min(x)
            miny = np.min(y)
            maxx = np.max(x)
            maxy = np.max(y)
            croped_mask = mask[miny:maxy, minx:maxx]
            if croped_mask.shape[0] * croped_mask.shape[1] < 15:
                # TODO choose another instance
                # print('Drop an instance')
                continue
            background_id = imgId if random.randint(1, 2) == 1 else random.choice(imgIds)  # copy-move or splicing
            background = coco.loadImgs(background_id)[0]
            background_h = background['height']
            background_w = background['width']

            mask_h = maxy - miny
            mask_w = maxx - minx
            if background_h < mask_h or background_w < mask_w:
                # TODO choose another background
                # print('Drop a background')
                continue
            new_mask = np.zeros(shape=(background_h, background_w), dtype=np.uint8)
            new_y = random.randint(0, background_h - mask_h)
            new_x = random.randint(0, background_w - mask_w)
            new_mask[new_y:new_y + mask_h, new_x:new_x + mask_w] = croped_mask
            new_mask = cv2.merge([new_mask, new_mask, new_mask])
            instance = img_data[miny:maxy, minx:maxx]  # instance with colored background
            croped_mask = cv2.merge([croped_mask, croped_mask, croped_mask])
            instance = cv2.bitwise_and(instance, croped_mask)  # instance with black background
            background_data = cv2.imread(os.path.join(img_path, background['file_name']), cv2.IMREAD_COLOR)

            if croped_mask.shape[0] * croped_mask.shape[1] < 1000 or random.randint(1, 5) == 1:  # just add
                # if croped_mask.shape[0]*croped_mask.shape[1]<1000:
                    # print('Use add because too small')
                # else:
                    # print('Use add because prob')
                new_background = cv2.bitwise_and(background_data, cv2.bitwise_not(new_mask))
                new_background[new_y:new_y + mask_h, new_x:new_x + mask_w] = cv2.add(new_background[new_y:new_y + mask_h, new_x:new_x + mask_w], instance)
            else:
                flag = np.random.choice([cv2.MONOCHROME_TRANSFER, cv2.NORMAL_CLONE, cv2.MIXED_CLONE], p=[.45, .45, .1])  # Poison blend
                new_background = cv2.seamlessClone(instance, background_data, croped_mask, ( new_x + (mask_w // 2), new_y + (mask_h // 2)), flags=flag) ## NORMAL_CLONE MIXED_CLONE
            new_file_name = os.path.join(manipulated_path, os.path.splitext(file_name)[0]) + '_f' + random.choice(['.png', '.jpg'])
            cv2.imwrite(new_file_name, new_background)
            cv2.imwrite(os.path.join(mask_path, os.path.splitext(file_name)[0]) + '_m.png', new_mask)

if __name__ == '__main__':
    main()
