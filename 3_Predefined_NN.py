
#Pretrained NN을 튜닝한다.

import torch

import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable



model = models.vgg16(pretrained=True)

print(model)
print("========네트워크 접근=========")
#각각을 다음과 같이 접근 ()이 이름
print(model.features)
print(model.avgpool)
print(model.classifier)


print("========잘라 보기=========")
print(model.classifier[0:1])


print("========특정 위치=========")
print(model.classifier[0])


print("========특정 클라시파이어 변경=========")
#배열 번호를 직접 입력한다.
model.classifier[0] = nn.Linear(10000, 100, bias = True)
print(model.classifier)


print("========맨 뒤에 모듈 추가=========")
my_layer1 = nn.Linear(1000, 3, bias=True)
model.classifier.add_module("7", my_layer1)   #스트링은 그냥 닉네임 같은거, 내부적으로는 정수로 변환되는 듯
print(model.classifier)
print(model.classifier[7])  #위에서 스트링으로 넣으면 정수형 넣으라고 함


print("========맨 뒤에 모듈 추가=========")
my_layer2 = nn.Sigmoid()
model.classifier.add_module("10", my_layer2)
print(model.classifier)
print(model.classifier[8])  #저 번호가 아닌 순서를 넣어야 함





#model = models.googlenet(num_classes=5, pretrained=True)
#output = model.features[:3](input)
#model2 = models.vgg16(pretrained=True)
#model = models.vgg19(pretrained=True).features = 100
