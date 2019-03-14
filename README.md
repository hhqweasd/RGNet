# RGNet

PyTorch implementation of RGNet [RGNet: Iterative Adversarial Inference with Re-inference Chain for Deep Graphical Models]. Submitted to ICICE.

<img src="./assets/1.PNG" width="100%">

## Requirements

- Python 3.6
- Pytorch 0.4+
- other: visdom, etc.

## CIFAR10 Results (Under the same or similar settings)
| Method | IS | FID |
|------|:-----:|------|
| MIX+WGAN | 4.04 | 70.58 | 
| Improved-GAN | 4.36 | 70.58 | 
| [AGE](https://github.com/sniklaus/pytorch-sepconv)| 4.96 | 64.61 |
| PixelIQN | 5.29 | 49.46 | 
| Dist-GAN | - | 45.60 | 
| ALI | 4.56 | 70.58 | 
| GibbsNet | 4.63 | 73.42 | 
| RGNet-1 | 5.21 | 61.88 |
| RGNet-2 | 5.51 | 56.85 |

## Usage

Need to modify the import file in ./models/ali.py when using different datasets.

Remember to modify the folder name of the saved files in ./ckpt or ./test when the train or test is finished.

Remember to modify the path when using ssl_lvdataset.py(SSL step 1), mlp_train.py(SSL step 2), is_example.py(IS), fid_example.py(FID)
Need tensorflow 1.4 to compute IS and FID

### Train
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

### Visualize (This part is annotated in train.py, but it works.)

- To visualize intermediate results and loss plots, run `python -m visdom.server` and go to the URL http://localhost:8097

### Output generated images
    # generate 50000 images
    $ python test_50k.py --test_count=500 --model=GibbsNet --repeat_generation=100 --is_train=1 --epoch=100 --dataset=SVHN --sampling_count=20
    $ python test_50k.py --test_count=500 --model=RGibbsNet --repeat_generation=100 --is_train=1 --epoch=100 --dataset=SVHN --sampling_count=20
    $ python test_50k.py --test_count=500 --model=GibbsNet --repeat_generation=100 --is_train=1 --epoch=200 --dataset=MNIST --input_channel=1 --width=28 --height=28
    $ python test_50k.py --test_count=500 --model=RGibbsNet --repeat_generation=100 --is_train=1 --epoch=200 --dataset=MNIST --input_channel=1 --width=28 --height=28
    $ python test_50k.py --test_count=500 --model=GibbsNet --repeat_generation=100 --is_train=1 --epoch=100 --dataset=CIFAR10 --sampling_count=20
    $ python test_50k.py --test_count=500 --model=RGibbsNet --repeat_generation=100 --is_train=1 --epoch=100 --dataset=CIFAR10 --sampling_count=20
    # image inpainting result
    $ python test_reconstruction.py --test_count=200 --model=RGibbsNet --is_train=1 --dataset=SVHN --epoch=100 --sampling_count=20 --batch_size=10

### MLP SSL Test
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

### IS and FID 
```
    $ python is_example.py
    $ python fid_example.py
```

## Implementation detail
- Following Adversarially learned inference.

## Code reference
Code references to https://github.com/wlwkgus/GibbsNet.git (ALI and GibbsNet)

Code references to https://github.com/bioinf-jku/TTUR.git (FID)

Code references to https://github.com/hhqweasd/WINN.git (IS)

## Author
(https://github.com/hhqweasd)
