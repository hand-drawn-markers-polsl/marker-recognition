import argparse

from make_models import make_simple_cnn
from train import train

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
    train(model, params)
