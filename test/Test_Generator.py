"""
Test_Generator.py
Will need to have both generator and upscaler models in same directory
"""

from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import math


catUpscaler = load_model('Cat_Upscaler.h5')
catGenerator = load_model('dcgan_cat_64_48.h5')

gen_amount = 4  # Amount of images that should be generated (this number should be a perfect square)
latent_size = catGenerator.input.shape[1]
upscale_factor = int(catUpscaler.output.shape[1] / catGenerator.output.shape[1])


def generate(gen_amount):  # Generation function
  noise_input = np.random.uniform(-1.0, 1.0, size=[gen_amount, latent_size])
  images = catGenerator.predict(noise_input)
  new_images = catUpscaler.predict(images)
  plt.figure(figsize=(22.2, 22.2))
  num_images = new_images.shape[0]
  image_size = new_images.shape[1]
  rows = int(math.sqrt(noise_input.shape[0]))
  for i in range(num_images):
    plt.subplot(rows, rows, i + 1)
    plt.imshow(np.array(new_images[i - 1]).reshape((64*upscale_factor, 48*upscale_factor, 3)))
    plt.axis('off')
  plt.show()
 

generate(gen_amount)
