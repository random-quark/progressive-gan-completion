import random
import numpy as np

latents = np.random.RandomState(1000).randn(
    1000, 512)  # 1000 random latents

print(len(latents[0]))

latents = latents[[477, 56, 83, 887, 583, 391,
                   86, 340, 341, 415]]  # hand-picked top-10


# x = [[1, 2, 3], [4, 5, 6]]
x = latents[[0]]

print(x)


# [1000, 512]
