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

# for image, label in data_loader:
#     print(np.shape(label))
testlv=torch.Tensor(0,0)
# trainlv=trainlv.numpy()
# trainlv = torch.Tensor(50000, 64)
testlabel = torch.Tensor(26032)
# trainlabel = np.zeros([50000, 10], np.int64)
# trainlabel = torch.from_numpy(trainlabel)

# print('trainlv', np.shape(trainlv))
# trainlabel=[]
# print('trainlabel', np.shape(trainlabel))

for i, data in enumerate(data_loader):
    test_dir = os.path.join(opt.test_dir, opt.model)
    # if i >= opt.test_count:
    #     break
    # cifar_dir = '/home/lzh/cifar'
    # misc.imsave(cifar_dir + '/' + 'real_{}.png'.format(i), np.transpose(data[0][0].numpy(), [1, 2, 0]))
    # print(i)
    (inputs, labels) = data
    for ii, label in enumerate(labels):
        testlabel[i*opt.batch_size+ii] = label
        # trainlabel[i*opt.batch_size+ii, label] = 1
        # print(trainlabel[i*opt.batch_size+ii])
    print('i',i)
    model.set_input(data[0])
    model.test()
    visuals = model.get_visuals(sample_single_image=False)
    lv = Variable(model.get_lv())
    # trainlv=np.concatenate([trainlv,lv.numpy()],0)
    testlv=torch.cat((testlv,lv),0)
    print(np.shape(testlv))
    # for iii,lvdata in enumerate(lv):
    #     print(type(lv))
    #     print(np.shape(lv))
    #     print(np.shape(trainlv))
    #     # trainlv[i*opt.batch_size+iii] = lvdata

    #
    # for j in range(opt.batch_size):
    #     np_image = visuals['fake_x'][j]
    #     misc.imsave(test_dir + '/' + 'fake_{}_{}.png'.format(i, j), np_image)

import pickle
save_file=open("test_g_epoch100.bin","wb")
pickle.dump(testlv,save_file)
pickle.dump(testlabel,save_file)
save_file.close()
