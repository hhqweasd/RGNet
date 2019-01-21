from data.data_loader import get_data_loader
from models.models import create_model
from option_parser import TestingOptionParser
from scipy import misc
import numpy as np
import torch
import os

parser = TestingOptionParser()
opt = parser.parse_args()
opt.batch_size = opt.repeat_generation
opt.gpu_ids = []

data_loader = get_data_loader(opt)

model = create_model(opt)
total_steps = 0
model.load(opt.epoch)
Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.FloatTensor
single_input = Tensor(
    1,
    opt.input_channel,
    opt.height,
    opt.width
)
repeated_input = Tensor(
    opt.batch_size,
    opt.input_channel,
    opt.height,
    opt.width
)

for i, data in enumerate(data_loader):
    print(i)
    print(np.shape(data[0][0]))
    for ii in range(3):
        for jj in range(32):
            for kk in range(16):
                data[0][0][ii,jj,kk+16]=0
    test_dir = os.path.join(opt.test_dir, opt.model)
    if i >= opt.test_count:
        break
    single_input.copy_(
        data[0][0].view(
            1,
            opt.input_channel,
            opt.height,
            opt.width
        )
    )
    np.shape(single_input)
    # misc.imsave(test_dir + '/' + 'real_{}.png'.format(i), np.transpose(single_input.numpy(), [1, 2, 0]))
    repeated_input.copy_(
        single_input.repeat(opt.batch_size, 1, 1, 1)
    )

    model.ali_model.set_input(repeated_input)
    visuals = model.reconstruction()

    for j in range(opt.batch_size):
        np_image = visuals['reconstruct_x'][j]
        misc.imsave(test_dir + '/' + 'reconstruct_{}_{}.png'.format(i, j), np_image)
        np_image = visuals['real_x'][j]
        misc.imsave(test_dir + '/' + 'real_{}_{}.png'.format(i, j), np_image)
