
import os
import time
import tensorflow as tf
from IPython import display

from generator import Generator, generator_loss
from discriminator import Discriminator, discriminator_loss
from images import generate_and_save_images
CHECKPOINT_DIR = '{}/../training_checkpoints'.format(os.path.dirname(os.path.realpath(__file__)))
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")

NUM_EXAMPLES_TO_GENERATE = 16

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

checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))

def train(dataset, epochs, batch_size, noise_dim):
    random_vector_for_gen = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, noise_dim])

    for epoch in range(epochs):
        start = time.time()

        for images in dataset:
            noise = tf.random_normal([batch_size, noise_dim])

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
            display.clear_output(wait=False)
            generate_and_save_images(generator, epoch + 1, random_vector_for_gen)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = CHECKPOINT_PREFIX)

        print ('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print ('Training Error - \tGenerator: {}\tDiscriminator: {}'.format(gen_loss, disc_loss))

    display.clear_output(wait=False)
    generate_and_save_images(generator, epochs, random_vector_for_gen)

    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
