from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
tf.enable_eager_execution()

import numpy as np

from train import train
from images import generate_gif, display_image

BUFFER_SIZE = 60000
BATCH_SIZE = 512 

EPOCHS = 150
NOISE_DIM = 100

# Retrieve the datasets
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5   # Normalize to [-1, 1]

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train(train_dataset, EPOCHS, BATCH_SIZE, NOISE_DIM)

display_image(EPOCHS)

generate_gif('dcgan.gif')
