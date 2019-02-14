import pickle
import numpy as np
import tensorflow as tf
import PIL
import PIL.Image as Image
import scipy.misc

tf.InteractiveSession()

ITERATIONS = 1

with open('/results/005-pgan-customimages-preset-v2-1gpu-fp32/network-snapshot-007320.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

z = np.random.randn(1, 512)
labels = np.zeros([z.shape[0]] + Gs.input_shapes[1][1:])

y = scipy.misc.imread('/source/bayeux1.png', mode='RGB').astype(np.float)
# y = Image.open('/source/bayeux1.png')

print(y)

# make mask
# mask = np.ones(y.shape)
# mask[128:, :, :] = 0.0


def save_image(image):
    images = np.array([image])
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0,
                     255.0).astype(np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
    PIL.Image.fromarray(images[0], 'RGB').save('img-sample.png')


for iteration in range(ITERATIONS):
    Gz = Gs.run(z, labels)
    generated_image = Gz[0]

    # y shape (256, 256, 3)
    # generated shape is (3, 256, 256)

    # reshaped_y = np.rollaxis(y, 1) # (3, 256, 256)
    # reshaped_y = np.rollaxis(y, 0, 3) # (256, 3, 256)

    # reshaped_y = y
    reshaped_y = y.transpose(2, 0, 1)

    reshaped_y = (reshaped_y / 255) * 2.0 - 1.0

    # images = Gz

    # MASK
    # contextual_loss = tf.reduce_sum(
    #     tf.contrib.layers.flatten(tf.abs(tf.mul(mask, Gz), tf.mul(mask, y))), 1)

    # print("Y SHAPE", y.shape)

    print("GEN", generated_image)
    print("RESH_Y", reshaped_y)

    # NO MASK
    difference = tf.abs(generated_image - reshaped_y)
    contextual_loss = tf.reduce_sum(
        tf.contrib.layers.flatten(difference), 1)

    output_image = tf.Session().run(difference)

    save_image(output_image)

    loss_out = tf.Session().run(contextual_loss)
    print(loss_out)

    # complete_image = tf.add(tf.mul(inverted_mask, Gz), (mask, y))
    # TODO: try adding the completed image to discriminator, not just generated image

    # perceptual_loss = D(Gz)

    # complete_loss = contextual_loss + 0.1 * perceptual_loss
    # print("contextual loss", contextual_loss)
    # complete_loss = contextual_loss

    # grad_complete_loss = tf.gradients(complete_loss, z)
