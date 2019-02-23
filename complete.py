# based on https://github.com/bamos/dcgan-completion.tensorflow
# and tutorial http://bamos.github.io/2016/08/09/deep-completion/

import pickle
import numpy as np
import tensorflow as tf
import PIL
import PIL.Image as Image
import scipy.misc

ITERATIONS = 200
LEARNING_RATE = 0.1


def save_image(image):
    global counter
    counter += 1
    # if counter % 50 > 0:
    #     return
    images = np.array([image])
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0,
                     255.0).astype(np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
    PIL.Image.fromarray(images[0], 'RGB').save(
        'img-sample-' + str(counter) + '.png')


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

y = scipy.misc.imread('/source/bayeux1.png', mode='RGB').astype(np.float)
# RESHAPE TEST IMAGE
# y shape (256, 256, 3)
# generated shape is (3, 256, 256)
reshaped_y = y.transpose(2, 0, 1)
reshaped_y = (reshaped_y / 255) * 2.0 - 1.0

# make mask
mask = np.zeros(reshaped_y.shape, dtype=np.float32)
for x in range(128):
    value = 1 - (x/128)
    mask[:, :, x] = value

# GENERATE IMAGE USING G() AND SAVE
Gz_tensor = Gs.get_output_for(tensorZ, labels)
Gz_tensor_image = Gz_tensor[0]

# MASK RIGHT HAND SIDE OF IMAGES BEFORE CALCULATING DIFFERENCE
# TODO: try adding the completed image to discriminator, not just generated image

# CALCULATE CONTEXTUAL LOSS (DIFFERENCE)
# MASK
contextual_loss = tf.reduce_sum(
    tf.contrib.layers.flatten(tf.abs(tf.multiply(mask, Gz_tensor_image) - tf.multiply(mask, reshaped_y))))

# PERCEPTUAL LOSS - GET D() SCORES FOR OUTPUT OF G()
# scores, _ = D.get_output_for(Gz_tensor)
# perceptual_loss = scores[0] * -1

# CALCULATE COMPLETE LOSS - perceptual + contextual
# complete_loss = contextual_loss + 0.1 * perceptual_loss
complete_loss = contextual_loss

optimizer = tf.train.AdamOptimizer()
optimize = optimizer.minimize(complete_loss, var_list=tensorZ)
sess.run(tf.variables_initializer(optimizer.variables()))
# grads_vars = optimizer.compute_gradients(complete_loss, var_list=tensorZ)
# apply = optimizer.apply_gradients(grads_vars)

for iteration in range(ITERATIONS):
    print("running iteration " + str(iteration))
    # test - print the complete loss
    complete_loss_out = sess.run(complete_loss)
    print("Complete loss", complete_loss_out)

    # OPTIMIZE Z BASED ON LOSS
    optimized = sess.run(optimize)

    # SAVE NEW IMAGE TO CHECK WHAT CHANGED AFTER OPTIMIZATION OF Z
    generated_image_pixels = sess.run(Gz_tensor_image)
    save_image(generated_image_pixels)

    # COMPLETE THE IMAGE WITH MASKS
    # complete_image = tf.add(tf.mul(inverted_mask, Gz), (mask, y))
