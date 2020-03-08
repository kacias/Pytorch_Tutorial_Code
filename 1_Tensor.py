import torch

import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


#=====================================
#그라디언트 이해

aa = torch.randn([4,3, 224, 224])
bb = torch.randn([4,3, 224, 224])
#print(aa.shape)


criterion = nn.MSELoss()

#직접하면 안된다

result = criterion(aa, bb)
d = torch.nn.functional.mse_loss (aa, bb)


print(result)
print(d)



#====================================
#shape 변환

cc = torch.randn([1,3,100,100])
print(cc.shape)

#view 전체 갯수를 보전하면서 shape 조절 (데이터 손실 x)
dd = cc.view([-1,1,100,100])
print(dd.shape)

#resize_ 강제로 특정 reshape 로 조절 (데이터 손실 O)
ee = cc.resize_([1,1,100,100])
print(ee.shape)

# ---------------------------------------------------------------
# 그라디언트 없는 텐서 생성 .detach()와 유사한 효과지만, 이건 처음 만들때부터 gradient 사용하지 않겠다고 선언
with torch.no_grad():
    no_gradient = torch.randn([1, 3, 100, 100])



#------------------------------
#텐서 형변환. 일반적으로 32bit를 쓴다. 
cc = cc.type('torch.DoubleTensor')
print(cc)
cc = cc.type('torch.FloatTensor')
print(cc)
