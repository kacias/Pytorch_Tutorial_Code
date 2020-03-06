# -*- coding: utf-8 -*-

import os

path = "D:/Pytorch_Project/UGATIT-pytorch-Seg_Network/dataset/wild2black/trainB/1"
file_list = os.listdir(path)

print ("file_list: {}".format(file_list))



f = open("image_list.txt", 'w')
for i in file_list:
    data = "%s\n" % i
    f.write(data)
f.close()


