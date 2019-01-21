from data.data_loader import get_data_loader
from models.models import create_model
from option_parser import GetMLPTestingOptionParser
from scipy import misc
import numpy as np
import torch
import os
from torchvision import datasets
from torch.autograd import Variable
parser = GetMLPTestingOptionParser()
opt = parser.parse_args()
opt.batch_size = opt.repeat_generation
opt.gpu_ids = []

data_loader = get_data_loader(opt)

model = create_model(opt)
total_steps = 0
model.load(opt.epoch)
Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.FloatTensor

testlv = torch.Tensor(0,0)
testlabel = torch.Tensor(10000)
iternum=0 # latent variable iterate number
for i, data in enumerate(data_loader):
    test_dir = os.path.join(opt.test_dir, opt.model)
    (inputs, labels) = data
    for ii, label in enumerate(labels):
        testlabel[i*opt.batch_size+ii] = label
    print('i', i)
    model.set_input(data[0])
    model.mlptest(iternum)
    visuals = model.get_visuals(sample_single_image=False)
    lv = Variable(model.get_lv())
    testlv=torch.cat((testlv, lv), 0)
    print(np.shape(testlv))

import pickle
save_file = open("test_rg_mnist_r1_i0_epoch300.bin", "wb")
pickle.dump(testlv,save_file)
pickle.dump(testlabel,save_file)
save_file.close()
