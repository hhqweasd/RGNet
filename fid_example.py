#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf

stats_path = './fid_stats_cifar10_train.npz' # training set statistics
inception_path = fid.check_or_download_inception(None) # download inception network

# load precalculated tr
# aining set statistics
f = np.load(stats_path)
mu_real, sigma_real = f['mu'][:], f['sigma'][:]
f.close()

# Paths
image_path = './test/HALI'
#image_path = './generate/ali/cifar10/{}'.format(i)
image_list = glob.glob(os.path.join(image_path, '*.png'))
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])

fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)

fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
print(" FID:", fid_value)

