import inception_score
import glob
import os
import numpy as np
from scipy.misc import imread
image_path = './test/HALI'
#image_path = './485'
image_list = glob.glob(os.path.join(image_path, '*.png'))
images = [imread(str(fn)).astype(np.float32) for fn in image_list]
print(inception_score.get_inception_score(images, splits=10))
