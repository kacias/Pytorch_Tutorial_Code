import os
import numpy as np
import cv2
from PIL import Image

original_path = './dataset/wild2black/trainA_seg/1/'
resized_path = './dataset/wild2black/trainA_seg/resize/'

file_list = os.listdir(original_path)
img_list = []

for item in file_list :
    if item.find('.jpg') is not -1 :
        img_list.append(item)

total_image = len(img_list)
index = 0

for name in img_list :

    img = Image.open('%s%s'%(original_path, name))
    img_array = np.array(img)
    img_resize = cv2.resize(img_array, (256,256), interpolation = cv2.INTER_AREA)
    img = Image.fromarray(img_resize)
    img.save('%s%s'%(resized_path, name))

    print(name + '   ' + str(index) + '/' + str(total_image))
    index = index + 1