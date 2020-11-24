import argparse

from make_models import make_simple_cnn
from train import train
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import matplotlib.cm as cm

from visualize import get_img_array, make_gradcam_heatmap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'name',
        type=str,
        help='Name of the experiment run, will be used to save models and\
              logs'
    )
    parser.add_argument(
        '-l',
        '--log',
        action='store_true',
        help='Enable tensorboard and model checkpoint logging to the log dir'
    )

    args = parser.parse_args()

    params = {
        'name': args.name,
        'img_directory': '../images/train',
        'img_size': (640, 640),
        'batch_size': 16,
        'epochs': 1,
        'validation_split': 0.2,
        'cross_validation': False,
        'tensorboard': args.log,
        'modelcheckpoint': args.log,
        'earlystopping': True,
    }

    model = make_simple_cnn()
    outputs = [layer.name for layer in model.layers]
    last_conv_layer_name = ""
    classifier_layer_names = []
    for output in reversed(outputs):
        if "conv2d" in output:
            last_conv_layer_name = output
            break
        classifier_layer_names.append(output)
    classifier_layer_names.reverse()
    print(last_conv_layer_name)
    print(classifier_layer_names)
    img = get_img_array("../images/train/true/IMG_6605.jpg", (640, 640))
    heatmap = make_gradcam_heatmap(
        img, model, last_conv_layer_name, classifier_layer_names
    )

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

    # We load the original image
    img = keras.preprocessing.image.load_img("../images/train/true/IMG_6605.jpg")
    img = keras.preprocessing.image.img_to_array(img)

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    save_path = "dupa.jpg"
    superimposed_img.save(save_path)

    train(model, params)
