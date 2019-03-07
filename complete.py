# based on https://github.com/bamos/dcgan-completion.tensorflow
# and tutorial http://bamos.github.io/2016/08/09/deep-completion/

import pickle
import numpy as np
import tensorflow as tf
import PIL
import PIL.Image as Image
import scipy.misc

from image_util import save_image

ITERATIONS = 50
LEARNING_RATE = 0.1

sess = tf.InteractiveSession()

counter = 0
v = 0
z = np.random.randn(1, 512)
tensorZ = tf.Variable(z)

init = tf.global_variables_initializer()
sess.run(init)

with open('/results/005-pgan-customimages-preset-v2-1gpu-fp32/network-snapshot-007320.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

labels = np.zeros([z.shape[0]] + Gs.input_shapes[1][1:])

# make gradient mask
mask = np.zeros((3, 256, 256), dtype=np.float32)
for x in range(128):
    value = 1 - (x/128)
    mask[:, :, x] = value


class Complete:
    def run(self, source_image, index):

        # GENERATE IMAGE USING G() AND SAVE
        Gz_tensor = Gs.get_output_for(tensorZ, labels)
        Gz_tensor_image = Gz_tensor[0]

        # CALCULATE CONTEXTUAL LOSS (DIFFERENCE)
        # WE MASK RIGHT HAND SIDE OF IMAGES BEFORE CALCULATING DIFFERENCE
        contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(tf.abs(tf.multiply(mask, Gz_tensor_image) - tf.multiply(mask, source_image))))

        # if counter == 0:
        #     save_image(mask, "mask", index)

        # PERCEPTUAL LOSS - GET D() SCORES FOR OUTPUT OF G()
        # TODO: try adding the completed image to discriminator, not just generated image
        scores, _ = D.get_output_for(Gz_tensor)
        perceptual_loss = scores[0] * -1

        # CALCULATE COMPLETE LOSS - perceptual + contextual
        complete_loss = contextual_loss + 0.1 * perceptual_loss

        optimizer = tf.train.AdamOptimizer()
        optimize = optimizer.minimize(complete_loss, var_list=tensorZ)
        sess.run(tf.variables_initializer(optimizer.variables()))

        for iteration in range(ITERATIONS):
            print("running iteration " + str(iteration))

            # OPTIMIZE Z BASED ON LOSS
            sess.run(optimize)

        generated_image_pixels = sess.run(Gz_tensor_image)
        # completed_image = tf.add(
        #     tf.mul(inverted_mask, generated_image_pixels), (mask, y))
        # completed_image_out = sess.run(completed_image)
        return generated_image_pixels
