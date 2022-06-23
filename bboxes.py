import cv2
import os

with open('shape_continuum/ProtoPNet/CUB_200_2011/CUB_200_2011/bounding_boxes.txt') as f:
    lines = f.readlines()
    image_info = dict()
    bbox = list()
    for l in lines:
        bbox_info = l.split()
        image_id = bbox_info[0]
        bbox.append(bbox_info[1]) # x
        bbox.append(bbox_info[2]) # y
        bbox.append(bbox_info[3]) # width
        bbox.append(bbox_info[4]) # height
        image_info[image_id] = bbox
    
    with open('shape_continuum/ProtoPNet/CUB_200_2011/CUB_200_2011/images.txt') as f2:
        image_paths = f2.readlines()
        main_path = 'shape_continuum/ProtoPNet/CUB_200_2011/CUB_200_2011/images/'
        for path in image_paths:
            img = path.split()
            img_dir = main_path + img[1]
            img_bbox_info = image_info[img[0]]
            x = int(float(img_bbox_info[0]))
            y = int(float(img_bbox_info[1]))
            width = int(float(img_bbox_info[2]))
            height = int(float(img_bbox_info[3]))

            image = cv2.imread(img_dir)
            cropped_image = image[y:y+height, x:x+width]
            target_path = os.path.join('shape_continuum/ProtoPNet/datasets/cub200_cropped', 'images')
            target_path = os.path.join(target_path, img[1].split('/')[0])

            if not os.path.exists(target_path):
                os.makedirs(target_path)

            cv2.imwrite(target_path +  '/' + img[0] + '_cropped.jpg', cropped_image)
