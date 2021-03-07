"""Visualize and explain the model."""

from math import ceil
from typing import List, Tuple
from pathlib import Path
import argparse
import os
import yaml

from tensorflow import keras
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from make_models import load_model


class FilterLayerVisualizer:
    """Visualizer of filters in convolutional models."""

    def __init__(self, model: keras.Model, layer_name: str, output_dir: Path):
        """Prepare visualizer for given model and convolutional layer.

        :param model: Model to be used.
        :param layer_name: Name of conv layer that will have its filters
            visualized. Layer name can be obtained from `mode.summary()`.
        :param output_dir: Path to directory where visualizations should be
            saved.
        """
        self._img_width, self._img_height = model.input_shape[1:3]
        self._layer_name = layer_name
        self._model_name = model.name
        self._layer = model.get_layer(name=layer_name)
        self._feature_extractor = keras.Model(
            inputs=model.inputs, outputs=self._layer.output
        )
        self._output_dir = output_dir

    def visualize(self, filter_num=8):
        """Perform the visualiztion and save results.

        :param filter_num: Number of filters to be viusalized. May be useful
            to limit this value if processing a very large layer.
        """
        all_imgs = []
        print('Computing max activations for filter:')
        for filter_index in range(filter_num):
            print(f'{filter_index}', end=', ', flush=True)
            img = self.visualize_filter(filter_index)
            all_imgs.append(img)
        print('\nFinished computing max activatons for filters.\n')

        filters_grid = self._make_filters_grid(all_imgs, filter_num=filter_num)
        self._save_vis(filters_grid)

    def visualize_filter(self, filter_index: int, iterations=30,
                         learning_rate=10.0) -> np.ndarray:
        """Visualize conv filter of given index."""
        filter_img = self._initialize_filter_img()
        for _ in range(iterations):
            loss, filter_img = self._gradient_ascent_step(
                filter_img, filter_index, learning_rate
            )

        filter_img = self._deprocess_image(filter_img[0].numpy())
        return filter_img

    def _initialize_filter_img(self) -> np.ndarray:
        """Init base random image to feed visualization algorithm."""
        img = tf.random.uniform(
            shape=(1, self._img_width, self._img_height, 3),
            dtype=float
        )
        return img

    def _make_filters_grid(self,
                           filters: List[np.ndarray],
                           filter_num=8,
                           cols=5,
                           grid_margin=5,
                           filter_margin=25) -> np.ndarray:
        """Create image with grid of filter visualizations.

        :param filters: List of images containing filter visualizations.
        :param filter_num: Limit number of filters to be displayed.
        :param cols: Number of columns inside the grid.
        :param grid_margin: Pixel width of margin between images in the grid.
        :param filter_margin: Width of margin which should be used to crop
            images inside the grid (useful to get rid of border artifacts).
        :return: Grid image with filter viusalizations.
        """
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
    def _gradient_ascent_step(self,
                              img: np.ndarray,
                              filter_index: int,
                              learning_rate: float,
                              ) -> Tuple[tf.Tensor, np.ndarray]:
        """Perform gradient step on particular filter for given image."""
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = self._compute_loss(img, filter_index)

        grads = tape.gradient(loss, img)
        grads = tf.math.l2_normalize(grads)
        img += learning_rate * grads
        return loss, img

    def _compute_loss(self, input_image: np.ndarray, filter_index: int,
                      margin=2) -> tf.Tensor:
        """Compute loss for feature extractor prediction."""
        activation = self._feature_extractor(input_image)
        # Omit borders on loss calculation, to avoid edge artifacts influence
        m = margin
        filter_activation = activation[:, m:-m, m:-m, filter_index]
        return tf.reduce_mean(filter_activation)

    def _save_vis(self, filters_grid) -> np.ndarray:
        """Save visualizations grid."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        name = self._model_name + '_' + self._layer_name + '_filter_max.png'
        keras.preprocessing.image.save_img(
            self._output_dir / name,
            filters_grid
        )

    @staticmethod
    def _init_img_grid(img_width: int, img_height: int, rows: int, cols: int,
                       margin=5) -> np.ndarray:
        """Prepare np array image which will contain visualizations grid.

        :param img_width: Width of a single image containing one filter
            viusalization.
        :param img_heght: Like above.
        :param rows: Number of rows in the grid.
        :param cols: Number of colums in the grid.
        :param margin: Grid margin between filter visualization images.
        :return: Image with size suitable to fit given grid, initally filled
            with zeros.
        """
        width = cols * img_width + (cols - 1) * margin
        height = rows * img_height + (rows - 1) * margin
        return np.zeros((height, width, 3))

    @staticmethod
    def _deprocess_image(img: np.ndarray, margin=25) -> np.ndarray:
        """Make given filter image displayable, apply given margin."""
        # Normalize array: center on 0.0, set variance to 0.15 (which is
        # somewhat arbitrary, but ensures that something is visible on
        # every image).
        img -= img.mean()
        img /= img.std() + keras.backend.epsilon()
        variance = 0.15
        img *= variance

        # Center crop
        img = img[margin:-margin, margin:-margin, :]

        # Prepare for displaying; change image range to 0.0 to 1.0
        img += 0.5
        img = np.clip(img, 0, 1)

        img *= 255
        img = np.clip(img, 0, 255).astype('uint8')
        return img


class IntermediateActivationsVisualizer:
    """Visualizer for activations of all layers in given model."""

    def __init__(self, model: keras.Model, output_dir: Path):
        """Init visualizer with given model and path to output directory."""
        i = self._last_non_dense_layer_index(model)
        self._model_name = model.name
        self._layer_outputs = [layer.output for layer in model.layers[:i]]
        self._layer_names = [layer.name for layer in model.layers[:i]]
        self._activation_model = keras.Model(
            inputs=model.input, outputs=self._layer_outputs)
        self._output_dir = output_dir

    def visualize(self, img: np.ndarray, name: str, cols=16):
        """Make the visualization for given image.

        :param img: Image which should fed to the network.
        :param name: Name which will be used for saving the visualization.
        :param cols: Number of columns in the activations grid image.
        """
        model_input = np.expand_dims(img, axis=0)
        activations = self._activation_model.predict(model_input)

        print('Computing intermediate activations for layer:')
        for layer_name, activation in zip(self._layer_names, activations):
            print(f'{layer_name}', end=', ', flush=True)
            grid_img = self.visualize_layer(activation, cols)
            self._save_vis(layer_name + name, grid_img)
        print('\nFinished computing intermediate activations.\n')

    def visualize_layer(self, layer_activation: np.ndarray,
                        cols: int) -> np.ndarray:
        """Visualize activations for given layer.

        :param layer_name: Name of layer which should have activations
            visualized.
        :param layer_activation: Array with activations for given layer.
        :param cols: Number of columns in the activations grid image.
        :return: image with grid of activations.
        """
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

        return display_grid

    def _save_vis(self, name: str, display_grid: np.ndarray):
        """Save layer activations visualization under given name."""
        os.makedirs(self._output_dir, exist_ok=True)
        fname = self._model_name + '_' + name + '_intermediate_activation.png'
        plt.imsave(
            self._output_dir / fname,
            display_grid,
            cmap='viridis'
        )

    @staticmethod
    def _last_non_dense_layer_index(model: keras.Model):
        """Get index of last non dense layer in the model."""
        i = 0
        for i, layer in enumerate(model.layers):
            if isinstance(layer, keras.layers.Dense):
                break
        return i - 1

    @staticmethod
    def deprocess_image(img):
        """Make given activations image displayable."""
        # Normalize array: center on 0.0, set variance to 64 (which is
        # somewhat arbitrary, but ensures that something is visible on
        # every image).
        img -= img.mean()
        img /= img.std() + keras.backend.epsilon()
        variance = 64
        img *= variance
        # Prepare for displaying; change image range to 0 to 255
        img += 128
        return np.clip(img, 0, 255).astype('uint8')


class ActivationsHeatmapVisualizer:
    """Visualizer for creating GradCAM based activations heatmaps."""

    def __init__(self, model: keras.Model, output_dir: Path):
        """Init visualizer for given model and output directory."""
        self._model_name = model.name
        self._output_dir = output_dir

        splitted_names = self._get_splitted_model_names(model)
        last_conv_layer_name, classifier_layer_names = splitted_names
        last_conv_layer = model.get_layer(last_conv_layer_name)

        self.last_conv_layer_model = keras.Model(
            model.inputs, last_conv_layer.output)

        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in classifier_layer_names:
            x = model.get_layer(layer_name)(x)
        self.classifier_model = keras.Model(classifier_input, x)

    def visualize(self, img: np.ndarray, name: str):
        """Make heatmap for image and save it using given name.

        Heatmap will be overlayed on the original image.
        """
        print('Computing GradCAM activations heatmap visualization.')
        model_input = np.expand_dims(img, axis=0)
        heatmap = self.make_gradcam_heatmap(model_input)

        heatmap = np.uint8(heatmap * 255)

        jet = cm.get_cmap('jet')

        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Overlay heatmap on the image
        overlay_intensity = 0.4
        overlayed_img = jet_heatmap * overlay_intensity + img * 255
        overlayed_img = keras.preprocessing.image.array_to_img(overlayed_img)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        fname = self._model_name + '_heatmap_' + name + '.png'
        print('Finished computing activations heatmap.\n')
        overlayed_img.save(self._output_dir / fname)

    def make_gradcam_heatmap(self, img_array: np.ndarray) -> np.ndarray:
        """Calculate gradcam heatmap for given image."""
        # Calculate top predicted class gradient in regard to the output
        # feature map of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output = self.last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            preds = self.classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # This is a vector where each entry is the mean intensity of the
        # gradient over a specific feature map channel
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted
        # class
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # Channel-wise mean
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        # Normalize between 0 and 1
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap

    @staticmethod
    def _get_splitted_model_names(model: keras.Model) -> Tuple[str, List[str]]:
        """plit given model in two parts: conv base and classifier.

        Returns name of last convolutional layer and tuple with all classifier
        layers names.
        """
        outputs = [layer.name for layer in model.layers]
        classifier_layer_names = []
        for output in reversed(outputs):
            if 'conv2d' in output:
                last_conv_layer_name = output
                break
            classifier_layer_names.append(output)
        classifier_layer_names.reverse()
        return last_conv_layer_name, classifier_layer_names


def main(params):
    """Run visualizer with given parameters."""
    vis_params = params['visualize']
    load_params = params['load']

    img_shape = (load_params['img_size'], load_params['img_size'])
    path = Path(vis_params['img_path'])
    img_name = path.stem
    img = keras.preprocessing.image.load_img(path, target_size=img_shape)
    img = keras.preprocessing.image.img_to_array(img)/255

    model = load_model('simple_regularized_cnn')
    model.summary()
    print()

    output_dir = Path('log/images/')

    FilterLayerVisualizer(
        model,
        vis_params['filter_layer_visualizer']['layer_name'],
        output_dir
    ).visualize()
    IntermediateActivationsVisualizer(
        model,
        output_dir/'intermediate/'
    ).visualize(img, img_name)
    ActivationsHeatmapVisualizer(
        model,
        output_dir
    ).visualize(img, img_name)


def _make_params() -> dict:
    """Make visualization parameters dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--name',
        nargs='?',
        type=str,
        help='Name of the model which will be used for visualization.\
              Overrides name from params.yaml.'
    )

    args = parser.parse_args()
    with open("params.yaml", 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)

    if args.name is not None:
        params['name'] = args.name

    return params


if __name__ == '__main__':
    PARAMS = _make_params()
    main(PARAMS)
