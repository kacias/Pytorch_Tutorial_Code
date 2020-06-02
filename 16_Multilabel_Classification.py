
# !/usr/bin/env python
# coding: utf-8




from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import time

from sklearn.metrics import confusion_matrix

from torchvision import datasets, transforms

import itertools

# from Parameters import gound_truth_list, answer_list

#from Util2 import plot_confusion_matrix

'''
import wandb
wandb.init(entity="wandb", project="pytorch-classification")
wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release
'''

gound_truth_list = []
answer_list = []
total_epoch = 1000
Leaning_Rate = 0.001

#아래 둘중 1개 선택
model_type ="mymodel"
#model_type ="VGG"


'''
#실험용 confusion matrix 
total_epoch = 20
Leaning_Rate = 0.01
'''


def imshow(inp, cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, cmap)


class Net(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50, bias=True)
        self.fc2 = nn.Linear(50, 9)



    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print("x shape1:{}".format(x.shape))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print("x shape2:{}".format(x.shape))

        x = x.view(x.size(0), -1)
        # print("x shape3:{}".format(x.shape))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return F.log_softmax(x, -1)


def fit(epoch, model, data_loader, phase='training', volatile=False, is_cuda=True):



    if model_type == "mymodel":

        # 내 모델 사용시
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        optimizer = optim.SGD(model.parameters(), lr=Leaning_Rate, momentum=0.5)

    elif model_type == "VGG":

        #VGG 사용시
        optimizer = optim.SGD(model.classifier.parameters(), lr=Leaning_Rate, momentum=0.5)


    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True

    running_loss = 0.0
    running_correct = 0

    for batch_idx, (data, target) in enumerate(data_loader):


        #print("is_cuda inside:{}".format(is_cuda))
        '''
        if is_cuda:
            data = data.to('cuda')
            target = target.to('cuda')
        else:
            data = data.to('cpu')
            target = target.to('cpu')
        '''

        # data, target = data.cuda(), target.cuda()

        #data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()


        # print("data shape:{}".format(data.shape))
        output = model(data)

        #-------------------------------------------------------


        #
        #--------------------------------------------------------

        #-------------------------------------------------------------------
        #자체 모델 사용 시
        if model_type == "mymodel":
            loss = F.nll_loss(output, target)
            running_loss += F.nll_loss(output, target, size_average=False).data

        # VGG 사용 시
        elif model_type == "VGG":
            loss = F.cross_entropy(output, target)
            running_loss += F.cross_entropy(output, target, size_average=False).data
        #----------------------------------------------------------------------

        preds = output.data.max(dim=1, keepdim=True)[1]

        gound_truth = target.data

        # print("preds:{}".format(preds))

        answer = preds.squeeze()

        # print("gound_truth:{}".format(gound_truth))
        # print("answer:{}".format(answer))

        a = gound_truth.data.detach().cpu().numpy()
        b = answer.data.detach().cpu().numpy()

        gound_truth_list.append(a)
        answer_list.append(b)

        # print("ground_truth numpy:{}".format(a))
        # print("answer numpy:{}".format(b))

        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct.item() / len(data_loader.dataset)
    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')

    # print("gound_truth_list:{}".format(gound_truth_list))
    # print("answer_list:{}".format(answer_list))

    return loss, accuracy


def training():



    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True

        print("cuda support")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("cpu support")
        torch.set_default_tensor_type('torch.FloatTensor')

    # ======================================================
    # 개 고양이
    # 폴더 구분 필요
    '''
    TRAIN_PATH = "./dogs-vs-cats_train"
    TEST_PATH = "./dogs-vs-cats_test"


    simple_transform = transforms.Compose([transforms.Resize((32, 32))
                                              , transforms.ToTensor()
                                              , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])
    train = ImageFolder(TRAIN_PATH, simple_transform)
    test = ImageFolder(TEST_PATH, simple_transform)

    '''
    # -----------------------------------------------------------------------------------------------
    # IM Data

    TRAIN_PATH = "./data/image_data_train_noise2"
    TEST_PATH = "./data/image_data_test_noise2"

    simple_transform = transforms.Compose([transforms.Resize((32, 32))
                                              , transforms.ToTensor()
                                              , transforms.Normalize((0.1307,), (0.3081,))])

    train = ImageFolder(TRAIN_PATH, simple_transform)
    test = ImageFolder(TEST_PATH, simple_transform)

    # ===============================================================================================
    # Mnist
    '''
    simple_transform = transforms.Compose([transforms.Resize((32, 32))
                                              , transforms.ToTensor()
                                              , transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST('../mnist_data/', train=True, transform = simple_transform, download=True)
    test = datasets.MNIST('../mnist_data/', train=False, transform = simple_transform, download=True)
    '''
    # ==============================================================================================

    print("len data1:{}".format(len(train)))
    print("len data2:{}".format(len(test)))

    print("class2idx:{}".format(train.class_to_idx))
    print("class:{}".format(train.classes))
    print("len:{}".format(len(train.classes)))

    train_data_loader = torch.utils.data.DataLoader(train, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(test, batch_size=64, num_workers=4, shuffle=False)

    print("------------- data load finished -------------------------")

    # wandb.watch(model, log="all")

    print("-------------- model selection------------------")
    #내가 만든 모델
    if model_type == "mymodel":

        model = Net(len(train.classes))

        if is_cuda:
            model.cuda()
            print("model send to cuda()")

    elif model_type == "VGG":


        #VGG 모델
        model = models.vgg16(pretrained=False)
        model = model
        print("model:{}".format(model) )
        model.classifier[6].out_features = 9


    #for param in model.features.parameters(): param.requires_grad = False

    graph_epoch = []
    train_losses = []
    train_accuracy = []
    val_losses = []
    val_accuracy = []

    print("is_cuda:{}".format(is_cuda))

    for epoch in range(1, total_epoch):

        print("-----------training: {} epoch-----------".format(epoch))

        epoch_loss, epoch_accuracy = fit(epoch, model, train_data_loader, phase='training')
        val_epoch_loss, val_epoch_accuracy = fit(epoch, model, valid_data_loader, phase='validation')

        graph_epoch.append(epoch)

        a = epoch_loss.detach().cpu().data.item()
        b = epoch_accuracy
        c = val_epoch_loss.detach().cpu().data.numpy()
        d = val_epoch_accuracy

        train_losses.append(a)
        train_accuracy.append(b)
        val_losses.append(c)
        val_accuracy.append(d)

        # print("train_losses:{}".format(train_losses))

        if epoch % 10 == 1:
            savePath = "./model/model_" + str(model_type) + str(epoch) + ".pth"
            torch.save(model.state_dict(), savePath)
            print("file save at {}".format(savePath))
            # wandb.save(savePath)


    # print("train_loss:{}".format(train_losses))

    # ---------------------------------------------
    # loss graph

    x_len = np.arange(len(train_losses))
    plt.plot(x_len, train_losses, marker='.', lw =1, c='red', label="train_losses")
    plt.plot(x_len, val_losses, marker='.', lw =1, c='cyan', label="val_losses")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.plot(x_len, val_accuracy, marker='.', lw =1, c='green', label="val_accuracy")
    plt.plot(x_len, train_accuracy, marker='.', lw =1, c='blue', label="train_accuracy")

    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

    # print("list:{}".format(gound_truth_list))
    # print("shape:{}".format(type(gound_truth_list)))


    gound_truth_list_1 = []
    for idx, data in enumerate(gound_truth_list):
        for j in data:
            gound_truth_list_1.append(j)

    print("gound truth list1:{}".format(gound_truth_list_1))

    # print("list2:{}".format(answer_list))
    # print("shape2:{}".format(type(answer_list)))

    ans_truth_list_1 = []
    for idx, data in enumerate(answer_list):
        for j in data:
            ans_truth_list_1.append(j)

    print("ans list2:{}".format(ans_truth_list_1))

    my_class = ["info", "Comm", "Enter", "News", "Edu", "Shop", "Fin", "Photo", "Navi"]

    # plot_confusion_matrix(gound_truth_list_1, ans_truth_list_1, classes=train.classes, normalize=True, title="Normalized Confusion Matrix")

    #plot_confusion_matrix(gound_truth_list_1, ans_truth_list_1, classes=my_class, normalize=True, title="Normalized Confusion Matrix")

    plt.show()


    # a= confusion_matrix(gound_truth_list_1, ans_truth_list_1)
    # print("confusion matrix: \n {}".format(a))


if __name__ == '__main__':
    training()
