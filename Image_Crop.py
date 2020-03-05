import os
from PIL import Image

resized_path = '~/Resized Training Image/'
cutted_path = '~/Cutted Training Image/'

img_list = os.listdir(resized_path)
total_image = len(img_list)
index = 1

for name in img_list :

    img = Image.open('%s%s'%(resized_path,name))
    cutted_img = img.crop((20,70,300,140))
    cutted_img.save('%s%s'%(cutted_path, name))

    print(name + '   ' + str(index) + '/' + str(total_image))
    index = index + 1