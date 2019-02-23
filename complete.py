# based on https://github.com/bamos/dcgan-completion.tensorflow
# and tutorial http://bamos.github.io/2016/08/09/deep-completion/

import pickle
import numpy as np
import tensorflow as tf
import PIL
import PIL.Image as Image
import scipy.misc

ITERATIONS = 2

sess = tf.InteractiveSession()

counter = 0


def save_image(image):
    global counter
    counter += 1
    images = np.array([image])
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0,
                     255.0).astype(np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
    PIL.Image.fromarray(images[0], 'RGB').save(
        'img-sample-' + str(counter) + '.png')


z = np.random.randn(1, 512)
tensorZ = tf.Variable(z)

init = tf.global_variables_initializer()
sess.run(init)

with open('/results/005-pgan-customimages-preset-v2-1gpu-fp32/network-snapshot-007320.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

labels = np.zeros([z.shape[0]] + Gs.input_shapes[1][1:])

v = 0

y = scipy.misc.imread('/source/bayeux1.png', mode='RGB').astype(np.float)

# make mask
# mask = np.ones(y.shape)
# mask[128:, :, :] = 0.0

# RESHAPE TEST IMAGE
# y shape (256, 256, 3)
# generated shape is (3, 256, 256)
reshaped_y = y.transpose(2, 0, 1)
reshaped_y = (reshaped_y / 255) * 2.0 - 1.0


for iteration in range(ITERATIONS):
    # GENERATE IMAGE USING G() AND SAVE
    Gz_tensor = Gs.get_output_for(tensorZ, labels)
    Gz_tensor_image = Gz_tensor[0]
    generated_image_pixels = sess.run(Gz_tensor_image)
    save_image(generated_image_pixels)

    # CALCULATE DIFFERENCE
    # WITH NO MASK
    # difference = tf.abs(generated_image - reshaped_y)
    # SAVE DIFFERENCE IMAGE
    # difference_image = tf.Session().run(difference)
    # save_image(difference_image)

    # MASK RIGHT HAND SIDE OF IMAGES BEFORE CALCULATING DIFFERENCE
    # TODO: this
    # TODO: try adding the completed image to discriminator, not just generated image

    # CALCULATE CONTEXTUAL LOSS (DIFFERENCE)
    # MASK
    # contextual_loss = tf.reduce_sum(
    #     tf.contrib.layers.flatten(tf.abs(tf.mul(mask, Gz), tf.mul(mask, y))), 1)
    # flattened_difference = tf.contrib.layers.flatten(difference)
    # flattened_difference_out = sess.run(flattened_difference)
    # print("Flattened difference", flattened_difference_out)
    # contextual_loss = tf.reduce_sum(flattened_difference)
    # contextual_loss_out = sess.run(contextual_loss)
    # print("Contextual loss", contextual_loss_out)

    # PERCEPTUAL LOSS - GET D() SCORES FOR OUTPUT OF G()
    scores, _ = D.get_output_for(Gz_tensor)
    image_scores = scores[0] * -1

    # CALCULATE COMPLETE LOSS
    # complete_loss = contextual_loss_out + 0.1 * pl
    # print("Complete loss", complete_loss)

    # GET GRADIENT OF LOSS
    grad_complete_loss_graph = tf.gradients(
        image_scores, tensorZ)
    # grad_complete_loss_graph = tf.gradients(complete_loss, z)
    grad_complete_loss = sess.run(grad_complete_loss_graph)
    loss_for_image = grad_complete_loss[0]

    # OPTIMIZE Z BASED ON LOSS
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    grads_vars = optimizer.compute_gradients(image_scores, var_list=tensorZ)
    apply = optimizer.apply_gradients(grads_vars)
    optimized = sess.run(apply)

    # SAVE NEW IMAGE TO CHECK WHAT CHANGED AFTER OPTIMIZATION OF Z
    generated_image_pixels = sess.run(Gz_tensor_image)
    save_image(generated_image_pixels)

    # COMPLETE THE IMAGE WITH MASKS
    # complete_image = tf.add(tf.mul(inverted_mask, Gz), (mask, y))
