import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import ceil


class FilterLayerVisualiser():
    def __init__(self, model, layer_name):
        self._img_width, self._img_height = model.input_shape[1:3]
        self._layer = model.get_layer(name=layer_name)
        self._feature_extractor = keras.Model(
            inputs=model.inputs, outputs=self._layer.output
        )

    def visualise(self, filter_num=8, cols=5, grid_margin=5, filter_margin=25):
        rows = ceil(filter_num/cols)
        all_imgs = []
        for filter_index in range(filter_num):
            print("Processing filter %d" % (filter_index,))
            loss, img = self.visualize_filter(filter_index)
            all_imgs.append(img)

        cropped_width = self._img_width - filter_margin * 2
        cropped_height = self._img_height - filter_margin * 2
        width = cols * cropped_width + (cols - 1) * grid_margin
        height = rows * cropped_height + (rows - 1) * grid_margin
        stitched_filters = np.zeros((height, width, 3))

        # Fill the picture with our saved filters
        grid_cell_width = cropped_width + grid_margin
        grid_cell_height = cropped_height + grid_margin
        for i in range(rows):
            for j in range(cols):
                if i * cols + j >= filter_num:
                    break
                img = all_imgs[i * cols + j]
                stitched_filters[
                    grid_cell_width * i:grid_cell_width * i + cropped_width,
                    grid_cell_height * j:grid_cell_height * j + cropped_height,
                    :,
                ] = img
        keras.preprocessing.image.save_img(
            "stiched_filters.png",
            stitched_filters
        )

    def compute_loss(self, input_image, filter_index, margin=2):
        activation = self._feature_extractor(input_image)
        # Omit borders on loss calculation, to avoid edge artifacts influence
        m = margin
        filter_activation = activation[:, m:-m, m:-m, filter_index]
        return tf.reduce_mean(filter_activation)

    @tf.function
    def gradient_ascent_step(self, img, filter_index, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = self.compute_loss(img, filter_index)
        # Compute gradients.
        grads = tape.gradient(loss, img)
        # Normalize gradients.
        grads = tf.math.l2_normalize(grads)
        img += learning_rate * grads
        return loss, img

    def initialize_image(self):
        img = tf.random.uniform(
            shape = (1, self._img_width, self._img_height, 3)
        )
        return img

    def visualize_filter(self, filter_index):
        # We run gradient ascent for 20 steps
        iterations = 30
        learning_rate = 10.0
        img = self.initialize_image()
        for iteration in range(iterations):
            loss, img = self.gradient_ascent_step(
                img, filter_index, learning_rate
            )

        # Decode the resulting input image
        img = self.deprocess_image(img[0].numpy())
        return loss, img

    def deprocess_image(self, img):
        # Normalize array: center on 0., ensure variance is 0.15
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.15

        # Center crop
        img = img[25:-25, 25:-25, :]

        # Clip to [0, 1]
        img += 0.5
        img = np.clip(img, 0, 1)

        # Convert to RGB array
        img *= 255
        img = np.clip(img, 0, 255).astype("uint8")
        return img

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
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def plot_heatmaps(img_path, img_size, model, save_path):
    img_array = get_img_array(img_path, size=img_size)
    last_conv_layer_name, classifier_layer_names = get_last_layer_names(model)
    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, classifier_layer_names
    )

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

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
    superimposed_img.save(save_path)
