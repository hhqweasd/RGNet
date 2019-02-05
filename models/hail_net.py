from models.hali import HALI
from models.base_model import BaseModel
from torch.autograd import Variable
import torch


class Hali(BaseModel):
    def __init__(self, opt):
        super(Hali, self).__init__(opt)

        self.hali_model = HALI(opt)
        self.opt = opt
        self.sampling_count = opt.sampling_count
        self.z = None
        self.fake_x = None
        self.z1 = None

        self.input = self.Tensor(
            opt.batch_size,
            opt.input_channel,
            opt.height,
            opt.width
        )
        self.x = None

    def forward(self, volatile=False):
        self.sampling()

        # clamped chain : HALI model
        self.hali_model.set_z(var=self.z)
        self.hali_model.set_input(self.x.data, is_z_given=True)

        self.hali_model.forward()

    def test(self):
        self.forward(volatile=True)

    def set_input(self, data):
        temp = self.input.clone()
        temp.resize_(self.input.size())
        temp.copy_(self.input)
        self.input = temp
        self.input.resize_(data.size()).copy_(data)

    def sampling(self, volatile=True):
        batch_size = self.opt.batch_size
        self.x = Variable(self.input)

        # unclamped chain
        if self.gpu_ids:
            self.z = Variable(torch.randn((batch_size, self.hali_model.encoder.k)).cuda())
        else:
            self.z = Variable(torch.randn((batch_size, self.hali_model.encoder.k)))

    def save(self, epoch):
        self.hali_model.save(epoch)

    def load(self, epoch):
        self.hali_model.load(epoch)

    def optimize_parameters(self, inferring_count=0):
        self.sampling()
        # clamped chain : HALI model
        self.hali_model.set_z(var=self.z)
        self.hali_model.set_input(self.x.data, is_z_given=True)

        self.hali_model.optimize_parameters()

    def get_losses(self):
        return self.hali_model.get_losses()

    def get_visuals(self, sample_single_image=True):
        return self.hali_model.get_visuals(sample_single_image=sample_single_image)

    def get_infervisuals(self, infernum, sample_single_image=True):
        return self.hali_model.get_infervisuals(infernum, sample_single_image=sample_single_image)

    def get_lv(self):
        return self.hali_model.get_lv()

    def remove(self, epoch):
        self.hali_model.remove(epoch)

    def reconstruction(self):
        return self.hali_model.reconstruction()
