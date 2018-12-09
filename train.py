from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
tf.enable_eager_execution()

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from IPython import display

CHECKPOINT_DIR = './training_checkpoints'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")

BUFFER_SIZE = 60000
BATCH_SIZE = 256

EPOCHS = 150
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

RAND_VECTOR_FOR_GEN = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

def get_image_name(epoch_no):
    return 'image_at_epoch{:04d}.png'.format(epoch_no)

def display_image(epoch_no):
    return PIL.IMAGE.open(get_image_name(epoch_no))

def generate_and_save_images(model, epoch, test_input):

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(get_image_name(epoch))
    plt.show()

def generate_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame= 2*(i**0.5)
            if round(frame) > roudn(last):
                last = frame
            else:
                continue

            image = imageio.imread(filename)
            writer.append_data(image)

        image = imageio.imread(filename)
        writer.append_data(image)

    os.system('cp {} {}.png'.format(filename, filename))

    display.Image(filename='{}.png'.format(filename))

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = tf.keras.layers.Dense(7*7*64, use_bias=False)

        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.conv2 = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.conv3 = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)

    def call(self, x, training=True):
        x = self.fc1(x)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.relu(x)

        x = tf.reshape(x, shape=(-1, 7, 7, 64))

        x = self.conv1(x)
        x = self.batchnorm2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.batchnorm3(x, training=training)
        x = tf.nn.relu(x)

        x = tf.nn.tanh(self.conv3(x))

        return x

class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1)

    def call(self, x, training=True):
        x = tf.nn.leaky_relu(self.conv1(x))
        x = self.dropout(x, training=training)
        x = tf.nn.leaky_relu(self.conv2(x))
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)

        return x

def discriminator_loss(real_output, generated_output):
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)

    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss

def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)

def train(dataset, epochs, noise_dim, checkpoint):

    for epoch in range(epochs):
        start = time.time()

        for images in dataset:
            noise = tf.random_normal([BATCH_SIZE, noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)

                real_output = discriminator(images, training=True)
                generated_output = discriminator(generated_images, training=True)

                gen_loss = generator_loss(generated_output)
                disc_loss = discriminator_loss(real_output, generated_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
            gradients_of_discrimantor = disc_tape.gradient(disc_loss, discriminator.variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discrimantor, discriminator.variables))

        if epoch % 1 == 0:
            display.clear_output(wait=True)
            generate_and_save_images(generator, epoch + 1, RAND_VECTOR_FOR_GEN)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = CHECKPOINT_PREFIX)

        print ('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, RAND_VECTOR_FOR_GEN)

# Retrieve the datasets
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5   # Normalize to [-1, 1]

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Initialize our generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Defun gives 10 sec/ epoch performance boost
generator.call = tf.contrib.eager.defun(generator.call)
discriminator.call = tf.contrib.eager.defun(discriminator.call)

# Setup our optimizers
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)
generator_optimizer = tf.train.AdamOptimizer(1e-4)

# Setup our checkpoint
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)

train(train_dataset, EPOCHS, NOISE_DIM, checkpoint)

checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))

display_image(EPOCHS)

generate_gif('dcgan.gif')
