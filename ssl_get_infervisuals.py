from data.data_loader import get_data_loader
from models.models import create_model
from option_parser import GetInferDisplayOptionParser
import torch
import os
from scipy import misc
import numpy as np
from PIL import Image
parser = GetInferDisplayOptionParser()
opt = parser.parse_args()
opt.gpu_ids = []

data_loader = get_data_loader(opt)

model = create_model(opt)
total_steps = 0
model.load(opt.epoch)
Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.FloatTensor

if opt.is_train == 1:
    test_dir = os.path.join(opt.test_dir, opt.model, opt.dataset, 'train', str(opt.epoch))
else:
    test_dir = os.path.join(opt.test_dir, opt.model, opt.dataset, 'test', str(opt.epoch))

print(test_dir)

infernum = 10 # infer display number
batch_input = Tensor(
    opt.batch_size,
    opt.input_channel,
    opt.height,
    opt.width
)
for i, data in enumerate(data_loader):
    if i>4:
        break
    # print(type(data[0][0]))
    # aa = data[0][0].numpy()*256
    # print(np.shape(aa))
    # print(type(aa))
    # print(aa)
    # import matplotlib.pyplot as plt
    # aa = aa.transpose((1,2,0))
    # print(np.shape(aa))
    # plt.imshow(aa)
    # im = Image.fromarray(aa.astype('uint8')).convert('RGB')
    # im.show()
    # input()
    batch_input.copy_(
        data[0].view(
            opt.batch_size,
            opt.input_channel,
            opt.height,
            opt.width
        )
    )
    # # print(type(batch_input))
    # # print(np.shape(batch_input))
    # aa = batch_input[0].numpy()
    # bb = batch_input[1].numpy()
    # # print(np.shape(aa))
    # # print(type(aa))
    # # print(aa)
    # import matplotlib.pyplot as plt
    # aa = aa.transpose((1,2,0))
    # bb = bb.transpose((1,2,0))
    # print(np.shape(aa))
    # plt.ion()
    # plt.figure(1)
    # plt.imshow(aa)
    # print(np.shape(bb))
    # plt.figure(2)
    # plt.imshow(bb)
    # # plt.show()
    # # plt.pause(5)
    # # plt.close()
    # input()

    model.set_input(batch_input)
    model.test()
    if opt.model == 'RGibbsNet':
        infervisuals = model.get_infervisuals(infernum, sample_single_image=False)
        for j in range(opt.batch_size):
            infer_image = infervisuals['infer_x'][j]
            real_image = infervisuals['real_x'][j]
            misc.imsave(test_dir + '/' + 'infer' + str(infernum) + '_{}_{}.png'.format(i, j), infer_image)
            misc.imsave(test_dir + '/' + 'real_{}_{}.png'.format(i, j), real_image)
    elif opt.model == 'GibbsNet':
        infervisuals = model.get_infervisuals(infernum)
        for j in range(opt.batch_size):
            infer_image = infervisuals['infer_x'][j]
            real_image = infervisuals['real_x'][j]
            misc.imsave(test_dir + '/' + 'infer' + str(infernum) + '_{}_{}.png'.format(i, j), infer_image)
            misc.imsave(test_dir + '/' + 'real_{}_{}.png'.format(i, j), real_image)