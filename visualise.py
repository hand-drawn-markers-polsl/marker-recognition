from tensorflow import keras
from math import ceil
import tensorflow as tf
import numpy as np
from make_models import load_model


class FilterLayerVisualiser():
    def __init__(self, model, layer_name):
        self._img_width, self._img_height = model.input_shape[1:3]
        self._layer = model.get_layer(name=layer_name)
        self._feature_extractor = keras.Model(inputs=model.inputs, outputs=self._layer.output)

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
        for i in range(rows):
            for j in range(cols):
                if i * cols + j >= filter_num:
                    break
                img = all_imgs[i * cols + j]
                stitched_filters[
                    (cropped_width + grid_margin) * i:(cropped_width + grid_margin) * i + cropped_width,
                    (cropped_height + grid_margin) * j:(cropped_height + grid_margin) * j + cropped_height,
                    :,
                ] = img
        keras.preprocessing.image.save_img("stiched_filters.png", stitched_filters)

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
        img = tf.random.uniform((1, self._img_width, self._img_height, 3))
        # TODO this scaling
        # ResNet50V2 expects inputs in the range [-1, +1].
        # Here we scale our random inputs to [-0.125, +0.125]
        return (img - 0.5) * 0.25

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


model = load_model('first_try_on_floats')
FilterLayerVisualiser(model, 'conv2d_1').visualise(filter_num=6)
