

#Pretrained NN이 항상 1이 나오도록 학습시킨다.
#Gound Truth를 1로 넣어주고 학습 시킨 후 모델 저장, 이후 test 해보면 1에 가까운 값이 나온다.
#하지만 1을 넘기는 값이 나오는 문제가 발생.
#Sigmoid를 맨 마지막에 추가하여 1로 맞추어 준다.
#학습 후 test를 해보면 모두 1로 바뀌어 있는 것을 알 수 있다.



import torch

import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchsummary import summary


#cuda로 보낸다
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print('========VGG 테스트 =========')
print("========입력데이터 생성 [batch, color, image x, image y]=========")
#이미지 사이즈를 어떻게 잡아도 vgg는 다 소화한다.



#==========================================
#1) 모델 생성
model = models.vgg16(pretrained=True).to(device)


print(model)




print("========맨 뒤에 모듈 추가=========")
my_layer2 = nn.Sigmoid()
model.classifier.add_module("7", my_layer2)


print(model)

print('========= Summary로 보기 =========')
#Summary 때문에 cuda, cpu 맞추어야 함
#뒤에 값이 들어갔을 때 내부 변환 상황을 보여줌
#adaptive average pool이 중간에서 최종값을 바꿔주고 있음
summary(model, (3, 100, 100))





#2) loss function
#꼭 아래와 같이 2단계, 클래스 선언 후, 사용
criterion = nn.MSELoss()

#3) activation function
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#모델이 학습 모드라고 알려줌
model.train()

#----------------------------
#epoch training
for i in range (100):

    #옵티마이저 초기화
    optimizer.zero_grad()

    #입력값 생성하고
    a = torch.randn(12,3,100,100).to(device)

    #모델에 넣은다음
    result = model(a)

    #결과와 동일한 shape을 가진 Ground-Truth 를 읽어서
    #target  = torch.randn_like(result)

    #타겟값을 1로 바꾸어서 네트워크가 무조건 1만 출력하도록 만든다.
    target  = torch.ones_like(result)


    #네트워크값과의 차이를 비교
    loss = criterion(result, target).to(device)

    #=============================
    #loss는 텐서이므로 item()
    print("epoch: {} loss:{} ".format(i, loss.item()))

    #loss diff값을 뒤로 보내서 grad에 저장하고
    loss.backward()

    #저장된 grad값을 기준으로 activation func을 적용한다.
    optimizer.step()


#print("=========== 학습된 파라미터만 저장 ==============")
#torch.save(model.state_dict(), 'trained_model.pt')


#모델이 바뀌었으므로 모델 전체를 저장
print("=========== 전체모델 저장 : VGG 처럼 모델 전체 저장==============")
torch.save(model, 'trained_model_all.pt')

