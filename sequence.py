import scipy.misc
import numpy as np
from complete import Complete
import PIL
import PIL.Image as Image

SEQUENCE_LENGTH = 1


def save_image(image, title, index):
    images = np.array([image])
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0,
                     255.0).astype(np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
    PIL.Image.fromarray(images[0], 'RGB').save(
        'img-' + str(index) + '-' + title + '.png')


complete = Complete()

# source_image shape (256, 256, 3)
# source_image = scipy.misc.imread('/source/bayeux1.png',
#                                  mode='RGB').astype(np.float)
source_image = scipy.misc.imread('/source/Black256.jpg',
                                 mode='RGB').astype(np.float)


# RESHAPE TEST IMAGE
# source_image shape (256, 256, 3)
# generated shape is (3, 256, 256)
reshaped_source_image = source_image.transpose(2, 0, 1)
reshaped_source_image = (reshaped_source_image / 255) * 2.0 - 1.0


def shift_image(source):
    shifted_image = source.copy()
    for channel in range(3):
        for y in range(256):
            for x in range(128):
                shifted_image[channel, y,
                              x] = source[channel, y, x + 128]
    shifted_image[:, :, 128:] = 0
    return shifted_image


def complete_image(counter, source):
    if (counter == 0):
        print("End of sequence - stop")
        return
    print("Executing sequence index", counter)
    counter -= 1
    image_index = SEQUENCE_LENGTH - counter
    save_image(source, '-i-source', image_index)
    shifted_source = shift_image(source)
    save_image(shifted_source, '-ii-shifted source', image_index)
    completed = complete.run(shifted_source, image_index)
    save_image(completed, '-iii-completed image', image_index)
    complete_image(counter, completed)


# first_image = shift_image(reshaped_source_image)
complete_image(SEQUENCE_LENGTH, reshaped_source_image)
