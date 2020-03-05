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
