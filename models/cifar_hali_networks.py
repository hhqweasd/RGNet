from collections import OrderedDict

import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from torch.optim import lr_scheduler

from utils.utils import layer_wrapper
from functools import partial, wraps


def normal_weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


class GANLoss(nn.Module):
    def __init__(self, use_gpu):
        super(GANLoss, self).__init__()
        self.use_gpu = use_gpu

    def __call__(self, input, target_value):
        loss_function = nn.BCELoss()
        target_tensor = torch.FloatTensor(input.size()).fill_(target_value)
        if self.use_gpu:
            return loss_function(input, Variable(target_tensor, requires_grad=False).cuda())
        else:
            return loss_function(input, Variable(target_tensor, requires_grad=False))


class VariationalEncoder(nn.Module):
    def __init__(self, k=64, gpu_ids=list()):
        super(VariationalEncoder, self).__init__()
        self.k = k
        self.gpu_ids = gpu_ids
        self.layer_wrapper = partial(
            layer_wrapper,
        )

        # Input: x:3x64x64 Output: z1:64x16x16  parameters:0.3M
        xz1 = [
            layer_wrapper(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=32,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                )
            ),
            layer_wrapper(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                )
            ),
            layer_wrapper(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1
                )
            ),
            layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                )
            ),
            layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(1, 1),
                )
            ),
            layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                )
            ),
            layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                ),
                norm_layer=None,
            ),
        ]
        self.xz1 = nn.Sequential(
            *xz1
        )
        
        # Input: z1:64x16x16 Output: z2:64x1x1 parameters:2.8M
        self.z1z2 = nn.Sequential(*[
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=(1, 1),
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=(4, 4),
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=(1, 1),
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=self.k*2,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                ),
                norm_layer=None,
                activation_function=None
            ),
        ])

    def forward(self, x):
        batch_size = x.size()[0]
        if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
            def model_encoder(x):
                z1 = self.xz1(x)
                z2_2 = torch.squeeze(self.z1z2(z1))
                u = z2_2[..., :self.k ]
                sigma = z2_2[..., self.k :]
                z2 = u + Variable(torch.randn([batch_size, self.k]).cuda()) * sigma
                return z1, z2
            return nn.parallel.data_parallel(model_encoder, x, self.gpu_ids)
        else:
            z1 = self.xz1(x)
            z2_2 = torch.squeeze(self.z1z2(z1))
            u = z2_2[..., :self.k ]
            sigma = z2_2[..., self.k :]
            z2 = u + Variable(torch.randn([batch_size, self.k])) * sigma
            return z1, z2
        
class VariationalDecoder(nn.Module):
    def __init__(self, k=64, gpu_ids=list()):
        self.k = k
        self.gpu_ids = gpu_ids
        super(VariationalDecoder, self).__init__()
        self.layer_wrapper = partial(
            layer_wrapper,
        )

        # Input: z2:64x1x1 Output: z1:64x16x16 
        self.z2z1 = nn.Sequential(*[
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=self.k,
                    out_channels=512,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                )
            ),
            self.layer_wrapper(
                nn.ConvTranspose2d(
                    in_channels=512,
                    out_channels=256,
                    kernel_size=(4, 4),
                    stride=(1, 1)
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=(1, 1),
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                )
            ),
            self.layer_wrapper(
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1,
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                )
            ),
            self.layer_wrapper(
                nn.ConvTranspose2d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                ),
                norm_layer=None,
            ),
        ])


        # Input: z1:64x16x16 Output: x:3x32x32
        self.z1x = nn.Sequential(*[
            self.layer_wrapper(
                nn.ConvTranspose2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(1, 1),
                )
            ),
            self.layer_wrapper(
                nn.ConvTranspose2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                )
            ),
            self.layer_wrapper(
                nn.ConvTranspose2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                )
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=3,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                ),
                norm_layer=None,
                activation_function=nn.Sigmoid()
            ),
        ])


    def forward(self, z):
        z = z.view([z.size()[0], z.size()[1], 1, 1])
        if self.gpu_ids and isinstance(z.data, torch.cuda.FloatTensor):
            def model_decoder(z):
                z1 = self.z2z1(z)
                x = self.z1x(z1)
                #print(x.data[0])
                return z1, x
            return nn.parallel.data_parallel(model_decoder, z, self.gpu_ids)
        else:
            z1 = self.z2z1(z)
            x = self.z1x(z1)
            return z1, x

class Discriminator(nn.Module):
    def __init__(self, k=64, gpu_ids=list()):
        super(Discriminator, self).__init__()
        self.k = k
        self.gpu_ids = gpu_ids
        self.layer_wrapper = partial(
            layer_wrapper,
            norm_layer=None,
        )

        # Input Format : NCHW
        # Input : 3 x 32 x 32
        # Output : 64 x 16 x 16
        self.d_x = nn.Sequential(*[
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=32,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                ),
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1
                ),
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                ),
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                ),
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                ),
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                ),
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                ),
            ),
        ])
        
        # Input : 128 x 16 x 16
        # Output : 256 x 1 x 1
        self.d_xz1 = nn.Sequential(*[
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                ),
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1
                ),
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                ),
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=(4, 4),
                    stride=(2, 2),
                    padding=1
                ),
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1
                ),
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=(4, 4),
                ),
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=(1, 1),
                ),
            ),
        ])

        # Input :  64+256 x 1 x 1
        # Output : 1
        self.d_xz1z2 = nn.Sequential(*[
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=self.k+512,
                    out_channels=1024,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                ),
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                ),
            ),
            self.layer_wrapper(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                ),
                activation_function=nn.Sigmoid()
            )
        ])

    def forward(self, x, z1, z2):
        if len(self.gpu_ids) > 0 and isinstance(x.data, torch.cuda.FloatTensor) and\
                isinstance(z1.data, torch.cuda.FloatTensor) and\
                isinstance(z2.data, torch.cuda.FloatTensor):
            def model(x, z1, z2):
                #import numpy as np
                #print('!!!!!!!1')
                #print(np.shape(x))
                d_x = self.d_x(x)
                #print(np.shape(d_x))
                #print(np.shape(z1))
                d_xz1_input = torch.cat((d_x, z1), 1)
                d_xz1 = self.d_xz1(d_xz1_input)
                #d_xz1 = torch.squeeze(self.d_xz1(d_xz1_input))
                z2 = z2.view(z2.size()[0], z2.size()[1], 1, 1)
                #import numpy as np
                #print(np.shape(d_x))
                #print(np.shape(d_xz1_input))
                #print(np.shape(d_xz1))
                #print(np.shape(z2))
                d_xz1z2_input = torch.cat((d_xz1, z2), 1)
                return torch.squeeze(self.d_xz1z2(d_xz1z2_input))
            return nn.parallel.data_parallel(model, (x, z1, z2), self.gpu_ids)
        d_x = self.d_x(x)
        d_xz1_input = torch.cat((d_x, z1), 1)
        d_xz1 = self.d_xz1(d_xz1_input)
        z2 = z2.view(z2.size()[0], z2.size()[1], 1, 1)
        d_xz1z2_input = torch.cat((d_xz1, z2), 1)
        return torch.squeeze(self.d_xz1z2(d_xz1z2_input))
