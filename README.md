# RGNet

PyTorch implementation of [Iterative Adversarial Inference with Re-inference Chain for Deep Graphical Models](https://www.jstage.jst.go.jp/article/transinf/E102.D/8/E102.D_2018EDL8256/_pdf). Paper is accepted by IEICE.

@article{liu2019iterative,
  title={Iterative Adversarial Inference with Re-Inference Chain for Deep Graphical Models},
  author={LIU, Zhihao and YIN, Hui and HUANG, Hua},
  journal={IEICE TRANSACTIONS on Information and Systems},
  volume={102},
  number={8},
  pages={1586--1589},
  year={2019},
  publisher={The Institute of Electronics, Information and Communication Engineers}
}

<img src="./assets/1.PNG" width="100%">

## Requirements （环境要求）

- Python 3.6
- Pytorch 0.4+
- other: visdom, etc.

## CIFAR10 Results (Under the same or similar settings) （CIFAR10 数据集结果）
| Method | IS | FID |
|------|:-----:|------|
| [AGE](https://github.com/DmitryUlyanov/AGE)| 4.96 | 64.61 |
| ALI | 4.56 | 70.58 | 
| GibbsNet | 4.63 | 73.42 | 
| RGNet-1 | 5.21 | 61.88 |
| RGNet-2 | 5.51 | 56.85 |

## Usage （使用方法）

Need to modify the import file in ./models/ali.py when using different datasets.

Remember to modify the folder name of the saved files in ./ckpt or ./test when the train or test is finished.

Remember to modify the path when using ssl_lvdataset.py(SSL step 1), mlp_train.py(SSL step 2), is_example.py(IS), fid_example.py(FID)
Need tensorflow 1.4 to compute IS and FID

### Train （训练）
- Train RGNet
```     
    $ python train.py --model=RGibbsNet --batch_size=100 --lr=1e-4 --dataset=CIFAR10 --gpu_ids=1 --sampling_count=20 --inferring_count=1 --epoch=100
    $ python train.py --model=RGibbsNet --batch_size=100 --lr=1e-4 --dataset=CIFAR10 --gpu_ids=1 --sampling_count=20 --inferring_count=2 --epoch=100
	
    $ python train.py --model=RGibbsNet --batch_size=100 --lr=1e-5 --dataset=SVHN --gpu_ids=1 --sampling_count=20 --inferring_count=1 --epoch=100
    $ python train.py --model=RGibbsNet --batch_size=100 --lr=1e-5 --dataset=SVHN --gpu_ids=1 --sampling_count=20 --inferring_count=2 --epoch=100
	
    $ python train.py --model=RGibbsNet --batch_size=100 --lr=1e-5 --dataset=MNIST --gpu_ids=1 --sampling_count=20 --inferring_count=1 --epoch=200
    $ python train.py --model=RGibbsNet --batch_size=100 --lr=1e-5 --dataset=MNIST --gpu_ids=1 --sampling_count=20 --inferring_count=2 --epoch=200
```
- Train GibbsNet
```
    $ python train.py --model=GibbsNet --batch_size=100 --lr=1e-4 --dataset=CIFAR10 --gpu_ids=1 --sampling_count=20 --epoch=100
    $ python train.py --model=GibbsNet --batch_size=100 --lr=1e-5 --dataset=SVHN --gpu_ids=1 --sampling_count=20 --epoch=100
    $ python train.py --model=GibbsNet --batch_size=100 --lr=1e-5 --dataset=MNIST --gpu_ids=1 --sampling_count=20 --epoch=300
```
-  Train ALI
```
    $ python train.py --model=GibbsNet --batch_size=100 --lr=1e-4 --dataset=CIFAR10 --gpu_ids=1 --sampling_count=0 --epoch=100
    $ python train.py --model=GibbsNet --batch_size=100 --lr=1e-5 --dataset=SVHN --gpu_ids=1 --sampling_count=0 --epoch=100
    $ python train.py --model=GibbsNet --batch_size=100 --lr=1e-5 --dataset=MNIST --gpu_ids=1 --sampling_count=0 --epoch=300
```

### Visualize (This part is annotated in train.py, but it works.) （训练过程可视化）

- To visualize intermediate results and loss plots, run `python -m visdom.server` and go to the URL http://localhost:8097

### Output generated images （生成图像）
    # generate 50000 images
    $ python test_50k.py --test_count=500 --model=GibbsNet --repeat_generation=100 --is_train=1 --epoch=100 --dataset=SVHN --sampling_count=20
    $ python test_50k.py --test_count=500 --model=RGibbsNet --repeat_generation=100 --is_train=1 --epoch=100 --dataset=SVHN --sampling_count=20
    $ python test_50k.py --test_count=500 --model=GibbsNet --repeat_generation=100 --is_train=1 --epoch=200 --dataset=MNIST --input_channel=1 --width=28 --height=28
    $ python test_50k.py --test_count=500 --model=RGibbsNet --repeat_generation=100 --is_train=1 --epoch=200 --dataset=MNIST --input_channel=1 --width=28 --height=28
    $ python test_50k.py --test_count=500 --model=GibbsNet --repeat_generation=100 --is_train=1 --epoch=100 --dataset=CIFAR10 --sampling_count=20
    $ python test_50k.py --test_count=500 --model=RGibbsNet --repeat_generation=100 --is_train=1 --epoch=100 --dataset=CIFAR10 --sampling_count=20
    # image inpainting result
    $ python test_reconstruction.py --test_count=200 --model=RGibbsNet --is_train=1 --dataset=SVHN --epoch=100 --sampling_count=20 --batch_size=10

### MLP SSL Test （半监督学习测试）
- CIFAR10
```
    $ python ssl_lvdataset.py --test_count=1 --model=GibbsNet --dataset=CIFAR10 --repeat_generation=250 --is_train=1 --epoch=100
    $ python ssl_lvdataset.py --test_count=1 --model=GibbsNet --dataset=CIFAR10 --repeat_generation=250 --is_train=0 --epoch=100
    $ python ssl_lvdataset.py --test_count=1 --model=RGibbsNet --dataset=CIFAR10 --repeat_generation=250 --is_train=1 --epoch=100
    $ python ssl_lvdataset.py --test_count=1 --model=RGibbsNet --dataset=CIFAR10 --repeat_generation=250 --is_train=0 --epoch=100
    $ python mlp_train.py
```
- SVHN
```
    $ python ssl_lvdataset.py --test_count=1 --model=GibbsNet --dataset=SVHN --repeat_generation=250 --is_train=1 --epoch=100
    $ python ssl_lvdataset.py --test_count=1 --model=GibbsNet --dataset=SVHN --repeat_generation=250 --is_train=0 --epoch=100
    $ python ssl_lvdataset.py --test_count=1 --model=RGibbsNet --dataset=SVHN --repeat_generation=250 --is_train=1 --epoch=100
    $ python ssl_lvdataset.py --test_count=1 --model=RGibbsNet --dataset=SVHN --repeat_generation=250 --is_train=0 --epoch=100
    $ python mlp_train.py
```
- MNIST
```
    $ python ssl_lvdataset.py --test_count=1 --model=GibbsNet --dataset=MNIST --repeat_generation=250 --is_train=1 --epoch=200
    $ python ssl_lvdataset.py --test_count=1 --model=GibbsNet --dataset=MNIST --repeat_generation=250 --is_train=0 --epoch=200
    $ python ssl_lvdataset.py --test_count=1 --model=RGibbsNet --dataset=MNIST --repeat_generation=250 --is_train=1 --epoch=200
    $ python ssl_lvdataset.py --test_count=1 --model=RGibbsNet --dataset=MNIST --repeat_generation=250 --is_train=0 --epoch=200
    $ python mlp_train.py
```

### IS and FID （IS和FID评估）
```
    $ python is_example.py
    $ python fid_example.py
```

## Implementation detail （实现细节）
- Following Adversarially learned inference.

## Code reference （代码参考）
Code references to https://github.com/wlwkgus/GibbsNet.git (ALI and GibbsNet)

Code references to https://github.com/bioinf-jku/TTUR.git (FID)

Code references to https://github.com/kjunelee/WINN.git (IS)

## Author （作者）
(https://github.com/hhqweasd)
