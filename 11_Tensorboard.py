
#=============================================================
#Tensorboard를 테스트 해 본다.
#설치
#pip install tensorboard
#pip install tensorboardX
#https://seongkyun.github.io/others/2019/05/11/pytorch_tensorboard/
#https://jangjy.tistory.com/343
#코드 빌드 후, 터미널에서 아래 실행 후 웹사이트 방문
#tensorboard --logdir = ./runs
#==============================================================


import torch

import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchsummary import summary


import matplotlib.pyplot as plt
import numpy as np

from tensorboardX import SummaryWriter

#1) 텐서보드 생성
tf_summary = SummaryWriter()

#cuda로 보낸다
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print('========VGG 테스트 =========')


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
#criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()
#criterion = nn.L1Loss()
#criterion = nn.Tanh()
#criterion = nn.KLDivLoss()

#criterion = nn.NLLLoss()
#criterion = nn.CrossEntropyLoss()
#criterion = nn.MarginRankingLoss()
#criterion = nn.CosineEmbeddingLoss()


#3) activation function
learning_rate = 1e-4
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)


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



    #================================
    #2) 텐서보드 값 입력
    tf_summary.add_scalar('loss/loss_a', loss.item(), i)
    tf_summary.add_scalar('learning_rate', learning_rate, i)




    #loss diff값을 뒤로 보내서 grad에 저장하고
    loss.backward()

    #저장된 grad값을 기준으로 activation func을 적용한다.
    optimizer.step()


#3) 텐서보드 꼭 닫기
tf_summary.close()

