from math import ceil
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from make_models import load_model


class FilterLayerVisualizer:
    def __init__(self, model, layer_name, output_dir):
        self._img_width, self._img_height = model.input_shape[1:3]
        self._layer_name = layer_name
        self._model_name = model.name
        self._layer = model.get_layer(name=layer_name)
        self._feature_extractor = keras.Model(
            inputs=model.inputs, outputs=self._layer.output
        )
        self._output_dir = output_dir

    def visualize(self, filter_num=8):
        all_imgs = []
        print('Computing max activations for filter:')
        for filter_index in range(filter_num):
            print(f'{filter_index}', end=', ', flush=True)
            _, img = self.visualize_filter(filter_index)
            all_imgs.append(img)
        print('\nFinished computing max activatons for filters.\n')

        filters_grid = self._make_filters_grid(all_imgs)
        self._save_vis(filters_grid)

    def visualize_filter(self, filter_index, iterations=30, lr=10.0):
        filter_img = self._initialize_filter_img()
        for _ in range(iterations):
            loss, filter_img = self._gradient_ascent_step(
                filter_img, filter_index, lr
            )

        filter_img = self._deprocess_image(filter_img[0].numpy())
        return loss, filter_img

    def _initialize_filter_img(self):
        img = tf.random.uniform(
            shape=(1, self._img_width, self._img_height, 3),
            dtype=float
        )
        return img

    def _make_filters_grid(self, filters, filter_num=8, cols=5,
                           grid_margin=5, filter_margin=25):
        rows = ceil(filter_num/cols)

        cropped_width = self._img_width - filter_margin * 2
        cropped_height = self._img_height - filter_margin * 2
        filters_grid = self._init_img_grid(
            cropped_width, cropped_height, rows, cols, grid_margin
        )

        grid_cell_width = cropped_width + grid_margin
        grid_cell_height = cropped_height + grid_margin

        for i in range(rows):
            for j in range(cols):
                if i * cols + j >= filter_num:
                    break

                filters_grid[
                    grid_cell_width * i:grid_cell_width * i + cropped_width,
                    grid_cell_height * j:grid_cell_height * j + cropped_height,
                    :,
                ] = filters[i * cols + j]

        return filters_grid

    @tf.function
    def _gradient_ascent_step(self, img, filter_index, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = self._compute_loss(img, filter_index)

        grads = tape.gradient(loss, img)
        grads = tf.math.l2_normalize(grads)
        img += learning_rate * grads
        return loss, img

    def _compute_loss(self, input_image, filter_index, margin=2):
        activation = self._feature_extractor(input_image)
        # Omit borders on loss calculation, to avoid edge artifacts influence
        m = margin
        filter_activation = activation[:, m:-m, m:-m, filter_index]
        return tf.reduce_mean(filter_activation)

    def _save_vis(self, filters_grid):
        os.makedirs(self._output_dir, exist_ok=True)
        keras.preprocessing.image.save_img(
            self._output_dir + self._model_name + '_' + self._layer_name +
            '_filter_max.png',
            filters_grid
        )

    @staticmethod
    def _init_img_grid(img_width, img_height, rows, cols, margin=5):
        width = cols * img_width + (cols - 1) * margin
        height = rows * img_height + (rows - 1) * margin
        return np.zeros((height, width, 3))

    @staticmethod
    def _deprocess_image(img, margin=25):
        # Normalize array: center on 0., ensure variance is 0.15
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.15

        # Center crop
        img = img[margin:-margin, margin:-margin, :]

        img += 0.5
        img = np.clip(img, 0, 1)

        img *= 255
        img = np.clip(img, 0, 255).astype("uint8")
        return img


class IntermediateActivationsVisualizer:
    def __init__(self, model, output_dir):
        i = last_non_dense_layer_index(model)
        self._model_name = model.name
        self._layer_outputs = [layer.output for layer in model.layers[:i]]
        self._layer_names = [layer.name for layer in model.layers[:i]]
        self._activation_model = models.Model(
            inputs=model.input, outputs=self._layer_outputs)
        self._output_dir = output_dir

    def visualize(self, img, cols=16):
        activations = self._activation_model.predict(img)

        print('Computing intermediate activations for layer:')
        for name, activation in zip(self._layer_names, activations):
            print(f'{name}', end=', ', flush=True)
            self.visualize_layer(name, activation, cols)
        print('\nFinished computing intermediate activations.\n')

    def visualize_layer(self, layer_name, layer_activation, cols):
        n_features = layer_activation.shape[-1]

        size = layer_activation.shape[1]
        rows = n_features // cols
        display_grid = np.zeros((size * rows, cols * size))

        for row in range(rows):
            for col in range(cols):
                channel_image = layer_activation[
                    0,
                    :, :,
                    row * cols + col
                ]
                channel_image = self.deprocess_image(channel_image)
                display_grid[
                    row * size:(row + 1) * size,
                    col * size:(col + 1) * size
                ] = channel_image

        self._save_vis(layer_name, display_grid)

    def _save_vis(self, name, display_grid):
        os.makedirs(self._output_dir, exist_ok=True)
        plt.imsave(
            self._output_dir + self._model_name + '_' + name +
            '_intermediate_activation.png',
            display_grid,
            cmap='viridis'
        )

    @staticmethod
    def deprocess_image(img):
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 64
        img += 128
        return np.clip(img, 0, 255).astype('uint8')


def get_img_array(img_path, size):
    img = load_img(img_path, target_size=size)
    array = img_to_array(img)
    return np.expand_dims(array, axis=0)


def get_last_layer_names(model):
    outputs = [layer.name for layer in model.layers]
    last_conv_layer_name = ""
    classifier_layer_names = []
    for output in reversed(outputs):
        if "conv2d" in output:
            last_conv_layer_name = output
            break
        classifier_layer_names.append(output)
    classifier_layer_names.reverse()
    return last_conv_layer_name, classifier_layer_names


def make_gradcam_heatmap(img_array, model, last_conv_layer_name,
                         classifier_layer_names):
    # First, we create a model that maps the input image
    # to the activations of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations
    # of the last conv layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input
    # image with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-5)
    return heatmap


def plot_heatmaps(img_path, img_size, model, save_path):
    img_array = get_img_array(img_path, size=img_size)
    last_conv_layer_name, classifier_layer_names = get_last_layer_names(model)
    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, classifier_layer_names
    )

    # We load the original image
    img = keras.preprocessing.image.load_img(img_path)
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
    os.makedirs(save_path, exist_ok=True)
    superimposed_img.save(save_path + model.name + '_heatmap.png')


def last_non_dense_layer_index(model):
    i = 0
    for i, layer in enumerate(model.layers):
        if isinstance(layer, keras.layers.Dense):
            break

    return i - 1


def main():
    path = '../images/train/true/IMG_20201101_133135.jpg'
    img = keras.preprocessing.image.load_img(
        '../images/train/true/IMG_20201101_133135.jpg'
    )
    """
    TODO: Below part which loads single img should be unified and
    placed in data loader module.
    """
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    model = load_model('simple_regularized_cnn')
    model.summary()
    print()

    plot_heatmaps(path, (256, 256), model, "log/images/")
    FilterLayerVisualizer(model, 'conv2d_3', 'log/images/').visualize()
    IntermediateActivationsVisualizer(model, 'log/images/').visualize(img)


if __name__ == '__main__':
    main()
