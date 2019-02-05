import argparse
import torch
import os

from utils import utils


class OptionParser(object):
    def __init__(self):
        print('=========================option_parse.py-->class OptionParse-->__init__(self)')
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def parse_args(self):
        print('=========================option_parse.py-->class OptionParse-->parse_args(self)')
        opt = self.parser.parse_args()
        str_ids = str(opt.gpu_ids)
        opt.gpu_ids = []
        for str_id in str_ids.split(','):
            opt.gpu_ids.append(int(str_id))
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        args = vars(opt)

        print('-------- [INFO] Options --------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))

        expr_dir = os.path.join(opt.ckpt_dir, opt.model)
        utils.mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(' [INFO] Options\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
        print('------------- END -------------')
        return opt

    def set_arguments(self):
        print('=========================option_parse.py-->class OptionParse-->set_arguments(self)')
        # training options
        self.parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of dataset. CIFAR10 default')
        self.parser.add_argument('--batch_size', type=int, default=100, help='batch size')
        self.parser.add_argument('--num_preprocess_workers', type=int, default=14, help='num preprocess workers')
        self.parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids ex) 0,1,2')
        self.parser.add_argument('--ckpt_dir', type=str, default='./ckpt/', help='checkpoint dir')
        self.parser.add_argument('--model', type=str, default='GibbsNet', help='name of model')
        self.parser.add_argument('--height', type=int, default=32, help='height')
        self.parser.add_argument('--input_channel', type=int, default=3, help='input channels')
        self.parser.add_argument('--width', type=int, default=32, help='width')
        self.parser.add_argument('--z_dimension', type=int, default=64, help='latent z dimension')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--sampling_count', type=int, default=0, help='gibbsnet sampling count')
        self.parser.add_argument('--epoch', type=int, default=100, help='epoch')
        # visualize options
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')


class TrainingOptionParser(OptionParser):
    def set_arguments(self):
        print('=========================option_parse.py-->class TrainingOptionParser-->set_arguments(self)')
        super(TrainingOptionParser, self).set_arguments()
        self.parser.add_argument('--is_train', type=int, default=1, help='is training')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.ckpt_dir]/[opt.model]/web/')
        self.parser.add_argument('--print_freq', type=int, default=2000, help='iteration count per a single print')
        self.parser.add_argument('--plot_freq', type=int, default=15000, help='iteration count per a single plot')
        self.parser.add_argument('--inferring_count', type=int, default=0, help='rgnet re-inferring count')


class TestingOptionParser(OptionParser):
    def set_arguments(self):
        print('=========================option_parse.py-->class TestingOptionParser-->set_arguments(self)')
        super(TestingOptionParser, self).set_arguments()
        self.parser.add_argument('--is_train', type=int, default=0, help='is training')
        self.parser.add_argument('--repeat_generation', type=int, default=10, help='repeat generation count per a single image')
        self.parser.add_argument('--test_count', type=int, default=20, help='test input images count')
        self.parser.add_argument('--test_dir', type=str, default='./test/', help='test dir')

    def parse_args(self):
        print('=========================option_parse.py-->class TestingOptionParser-->parse_args(self)')
        opt = self.parser.parse_args()
        str_ids = str(opt.gpu_ids)
        opt.gpu_ids = []
        for str_id in str_ids.split(','):
            opt.gpu_ids.append(int(str_id))
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        args = vars(opt)

        print('-------- [INFO] Options --------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))

        expr_dir = os.path.join(opt.ckpt_dir, opt.model)
        utils.mkdir(expr_dir)
        test_dir = os.path.join(opt.test_dir, opt.model)
        utils.mkdir(test_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(' [INFO] Options\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
        print('------------- END -------------')
        return opt


class GetMLPTestingOptionParser(OptionParser):
    def set_arguments(self):
        print('=========================option_parse.py-->class GetMLPTestingOptionParser-->set_arguments(self)')
        super(GetMLPTestingOptionParser, self).set_arguments()
        self.parser.add_argument('--is_train', type=int, default=1, help='is training')
        self.parser.add_argument('--repeat_generation', type=int, default=100, help='repeat generation count per a single image')
        self.parser.add_argument('--test_count', type=int, default=20000, help='test input images count')
        self.parser.add_argument('--test_dir', type=str, default='./inferimage/', help='test dir')

    def parse_args(self):
        print('=========================option_parse.py-->class GetMLPTestingOptionParser-->parse_args(self)')
        opt = self.parser.parse_args()
        str_ids = str(opt.gpu_ids)
        opt.gpu_ids = []
        for str_id in str_ids.split(','):
            opt.gpu_ids.append(int(str_id))
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        args = vars(opt)

        print('-------- [INFO] Options --------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))

        expr_dir = os.path.join(opt.ckpt_dir, opt.model)
        utils.mkdir(expr_dir)
        test_dir = os.path.join(opt.test_dir, opt.model)
        utils.mkdir(test_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(' [INFO] Options\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
        print('------------- END -------------')
        return opt


class GetInferDisplayOptionParser(OptionParser):
    def set_arguments(self):
        print('=========================option_parse.py-->class GetInferDisplayOptionParser-->set_arguments(self)')
        super(GetInferDisplayOptionParser, self).set_arguments()
        self.parser.add_argument('--is_train', type=int, default=1, help='is training')
        self.parser.add_argument('--test_dir', type=str, default='./inferimage/', help='test dir')
