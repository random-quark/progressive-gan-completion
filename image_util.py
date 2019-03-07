import PIL
import PIL.Image as Image
import numpy as np


def save_image(image, title, index):
    images = np.array([image])
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0,
                     255.0).astype(np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
    PIL.Image.fromarray(images[0], 'RGB').save(
        'img-' + str(index) + '-' + title + '.png')
