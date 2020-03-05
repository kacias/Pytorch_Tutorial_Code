
#Pretrained NN을 튜닝한다.

import torch

import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchsummary import summary


#cuda로 보낸다
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#----------------------------
#epoch training

#print("=========== 방법1: 모델을 정의하고 학습된 파라미터 읽어들이기 ==============")
#model = models.vgg16(pretrained=True).to(device)
#model.load_state_dict(torch.load('trained_model.pt'))


#print("=========== 방법2: 모델 전체 읽어들이기 ==============")
model = torch.load('trained_model_all.pt').to(device)



print('========= Summary로 보기 =========')
summary(model, (3, 100, 100))



print("========model weight 값 확인=========")
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)






#입력값 생성하고
a = torch.randn(12,3,100,100).to(device)

#평가 모드라고 알려줌
model.eval()

#모델에 넣은다음
result = model(a)

#결과를 확인해 본다.
print("result:{} ".format(result))







