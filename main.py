import argparse
import glob
import os

from make_models import make_simple_cnn, save_model
from train import train

from visualize import plot_heatmaps

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
        'batch_size': 8,
        'epochs': 100,
        'validation_split': 0.2,
        'cross_validation': False,
        'tensorboard': args.log,
        'modelcheckpoint': args.log,
        'earlystopping': True,
    }

    model = make_simple_cnn()
    train(model, params)
    save_model(model, "first_working_model")

    path_pattern = f"{params['img_directory']}/**/*.jpg"
    base_dir = "heatmaps"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    for i, img_path in enumerate(glob.glob(path_pattern, recursive=True)):
        print(f"Iteration: {i}")
        img_name = f"{base_dir}/{os.path.basename(img_path)}"
        plot_heatmaps(img_path, params['img_size'], model, img_name)
