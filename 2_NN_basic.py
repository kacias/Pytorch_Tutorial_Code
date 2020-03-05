import torch

import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


#==============================
#가장 간단한 네트워크

x = Variable(torch.randn(10, 5))


model = nn.Linear(5, 7)
criterion = nn.MSELoss()


print(x.shape)
y = model(x)
print(y.shape)

target  = torch.randn_like(y)
print("target:{}".format(target))


loss = criterion(target, y)


print(model.weight.grad) # None
loss.backward()
print(model.weight.grad) # Prints a 7x5 tensor


#================================



print(model)
