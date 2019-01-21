from data.data_loader import get_data_loader
from models.models import create_model
from option_parser import GetMLPTestingOptionParser
import numpy as np
import torch
import os
from torch.autograd import Variable
import pickle

parser = GetMLPTestingOptionParser()
opt = parser.parse_args()
opt.batch_size = opt.repeat_generation
opt.gpu_ids = []

data_loader = get_data_loader(opt)

model = create_model(opt)
total_steps = 0
model.load(opt.epoch)
Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.FloatTensor

testlv = torch.Tensor(0, 0)
if opt.is_train == 1:
    if opt.dataset == 'SVHN':
        testlabel = torch.Tensor(73257)
    elif opt.dataset == 'MNIST':
        testlabel = torch.Tensor(60000)
    elif opt.dataset == 'CIFAR10':
        testlabel = torch.Tensor(50000)
else:
    if opt.dataset == 'SVHN':
        testlabel = torch.Tensor(26032)
    elif opt.dataset == 'MNIST':
        testlabel = torch.Tensor(10000)
    elif opt.dataset == 'CIFAR10':
        testlabel = torch.Tensor(10000)

iternum = 0 # latent variable iterate number

for i, data in enumerate(data_loader):
    test_dir = os.path.join(opt.test_dir, opt.model)
    (inputs, labels) = data
    for ii, label in enumerate(labels):
        testlabel[i*opt.batch_size+ii] = label
    print('i', i)
    model.set_input(data[0])
    if opt.model == 'RGibbsNet':
        model.mlptest(iternum)
    elif opt.model == 'GibbsNet':
        model.test()
    lv = Variable(model.get_lv())
    testlv = torch.cat((testlv, lv), 0)
    print(np.shape(testlv))


if opt.model == 'RGibbsNet':
    if opt.is_train == 1:
        if opt.dataset == 'SVHN':
            save_file = open("train_rg_svhn_r01_i" + str(iternum) + "_epoch_" + str(opt.epoch) + ".bin", "wb")
        elif opt.dataset == 'MNIST':
            save_file = open("train_rg_mnist_r01_i" + str(iternum) + "_epoch_" + str(opt.epoch) + ".bin", "wb")
        elif opt.dataset == 'CIFAR10':
            save_file = open("train_rg_cifar_r012_i" + str(iternum) + "_epoch_" + str(opt.epoch) + ".bin", "wb")
    else:
        if opt.dataset == 'SVHN':
            save_file = open("test_rg_svhn_r012_i" + str(iternum) + "_epoch_" + str(opt.epoch) + "_2nd.bin", "wb")
        elif opt.dataset == 'MNIST':
            save_file = open("test_rg_mnist_r01_i" + str(iternum) + "_epoch_" + str(opt.epoch) + ".bin", "wb")
        elif opt.dataset == 'CIFAR10':
            save_file = open("test_rg_cifar_r012_i" + str(iternum) + "_epoch_" + str(opt.epoch) + ".bin", "wb")
elif opt.model == 'GibbsNet':
    if opt.is_train == 1:
        if opt.dataset == 'SVHN':
            save_file = open("train_g_svhn_epoch_" + str(opt.epoch) + "_lr5.bin", "wb")
        elif opt.dataset == 'MNIST':
            save_file = open("train_g_mnist_epoch_" + str(opt.epoch) + ".bin", "wb")
        elif opt.dataset == 'CIFAR10':
            save_file = open("train_g_cifar_epoch_" + str(opt.epoch) + ".bin", "wb")
    else:
        if opt.dataset == 'SVHN':
            save_file = open("test_g_svhn_epoch_" + str(opt.epoch) + "_lr5.bin", "wb")
        elif opt.dataset == 'MNIST':
            save_file = open("test_g_mnist_epoch_" + str(opt.epoch) + ".bin", "wb")
        elif opt.dataset == 'CIFAR10':
            save_file = open("test_g_cifar_epoch_" + str(opt.epoch) + ".bin", "wb")

pickle.dump(testlv, save_file)
pickle.dump(testlabel, save_file)
save_file.close()
