
#Pretrained NN을 튜닝한다.

import torch

import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchsummary import summary


#cuda로 보낸다
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


a = torch.randn(12,3,100,100).to(device)

print('========VGG 테스트 =========')
print("========입력데이터 생성 [batch, color, image x, image y]=========")
#이미지 사이즈를 어떻게 잡아도 vgg는 다 소화한다.
#중간에 adaptive average pooling



model = models.vgg16(pretrained=True).to(device)


print(model)
print('========= Summary로 보기 =========')
#Summary 때문에 cuda, cpu 맞추어야 함
#뒤에 값이 들어갔을 때 내부 변환 상황을 보여줌
#adaptive average pool이 중간에서 최종값을 바꿔주고 있음
summary(model, (3, 100, 100))


result = model(a)
print("result shape:{}".format(result.shape))

print("========model weight 값 측정=========")
'''
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)
'''

#결과와 동일한 shape을 가진 Ground-Truth 생성
target  = torch.randn_like(result)
print("target shape:{}".format(target.shape))

#loss function
#꼭 아래와 같이 2단계, 클래스 선언 후, 사용
criterion = nn.MSELoss()
loss = criterion(result, target).to(device)

#activation function
#옵티마이저
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



print("a shape:{}".format(a.shape))
optimizer.zero_grad()

#=============================
#loss는 텐서이므로 item()
print("loss:{} ".format(loss.item()))

#diff값을 뒤로 보내서 grad에 저장하고
loss.backward()

#저장된 grad값을 기준으로 activation func을 적용한다.
optimizer.step()


