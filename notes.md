input image y

generate a fake image G(z)

compare the non masked parts of g and y by doing simple subtraction - contextual loss
add masked y and inverse masked G(z) and get result from discriminator - perceptual loss

add losses together and multiply perceptual loss by lambda/0.1

use projected gradient descent to
