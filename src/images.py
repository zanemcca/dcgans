
import os
import matplotlib.pyplot as plt
import PIL
import imageio
import glob
from IPython import display

IMAGES_DIR = '{}/../images'.format(os.path.dirname(os.path.realpath(__file__)))

ON_HEADLESS_SERVER = not 'DISPLAY' in os.environ

def get_image_name(epoch_no):
    return '{}/image_at_epoch{:04d}.png'.format(IMAGES_DIR, epoch_no)

def display_image(epoch_no):
    return PIL.Image.open(get_image_name(epoch_no))

def generate_and_save_images(model, epoch, test_input):
    os.makedirs(IMAGES_DIR, exist_ok=True)

    predictions = model(test_input, training=False)

    # plt.close()
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(get_image_name(epoch))

    if not ON_HEADLESS_SERVER:
        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(0.001)

def generate_gif(ouput_filename):
    ouput_filename = '{}/{}'.format(IMAGES_DIR, ouput_filename)
    with imageio.get_writer(output_filename, mode='I') as writer:
        filenames = glob.glob('{}/image*.png'.format(IMAGES_DIR))
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

    os.system('cp {} {}.png'.format(ouput_filename, ouput_filename))

    if not ON_HEADLESS_SERVER:
        display.Image(filename='{}.png'.format(ouput_filename))
