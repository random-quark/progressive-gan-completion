import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import random

# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
with open('/results/005-pgan-customimages-preset-v2-1gpu-fp32/network-snapshot-007320.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

Gs(z)

print("INPUT")
print(*Gs.input_shapes[0][1:])

# Generate latent vectors.
latents = np.random.RandomState(1000).randn(
    1000, *Gs.input_shapes[0][1:])  # 1000 random latents

picks = np.random.random_integers(0, 999, 1)
print(picks)
latents = latents[picks]  # hand-picked top-10

# latents = np.random.choice(latents, 10)

# Generate dummy labels (not used by the official networks).
labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

# Run the generator to produce a set of images.
images = Gs.run(latents, labels)

# Convert images to PIL-compatible format.
images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0,
                 255.0).astype(np.uint8)  # [-1,1] => [0,255]
images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save('img%d.png' % idx)
