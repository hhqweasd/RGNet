from models.ali import ALI
from models.gibbs_net import GibbsNet
from models.rgibbs_net import RGibbsNet


def create_model(opt):
    print('=========================models.py-->create_model(opt)')
    if opt.model == 'ALI':
        return ALI(opt)
    elif opt.model == 'GibbsNet':
        return GibbsNet(opt)
    elif opt.model == 'RGibbsNet':
        return RGibbsNet(opt)
    else:
        raise Exception("This implementation only supports ALI, GibbsNet.")
