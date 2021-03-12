"""Train classification models."""

from datetime import datetime
from pathlib import Path
from typing import List
import argparse
import yaml

from tensorflow.data import Dataset
from tensorflow.keras import Model, callbacks, optimizers

from dataset_loading import load_train_dataset
from make_models import make_regularized_cnn


def train(train_ds: Dataset,
          val_ds: Dataset,
          model: Model,
          train_params: dict):
    """Train tensorflow model using given data.

    :param train_ds: Training dataset.
    :param val_ds: Validation dataset.
    :param model: Binary classifier model to train.
    :param train_params: Dict with training params. Must include keys:
        'lr': Learning rate of type float.
        'epochs': Number of training epochs of type int.
        'callbacks': Dict of callbacks params. See `make_callbacks` function
            docstring to examine required keys.
    """
    prepare_model(model, train_params['lr'])
    callbacks_list = make_callbacks(model.name, train_params['callbacks'])

    model.fit(
        train_ds,
        epochs=train_params['epochs'],
        callbacks=callbacks_list,
        validation_data=val_ds
    )


def prepare_model(model: Model, lr: float):
    """Prepare and compile binary classification model.

    :param model: Model to prepare.
    :param lr: Training learning rate.
    """
    model.compile(
        optimizer=optimizers.RMSprop(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )


def make_callbacks(name: str, cb_params: dict) -> List[callbacks.Callback]:
    """Make training callbacks list.

    Tensorboard driven data is stored in 'log' directory (which is currently
    not supervised by git/dvc). Model files (tracked by dvc) are stored in
    'data' directory.
    :param name: Name of given model/experiment. Used for saving logs and
        checkpoints.
    :param callbacks: Dict of callbacks params with keys:
        'tensorboard': Switch to enable tensorboard of type bool.
        'modelcheckpoint': Switch to enable model saving of type bool.
        'earlystopping': Switch to enable early stopping of type bool.
        'stopping_delta': Minimal delta to prevent stoppping of type float.
        'stopping_patience': Number of epochs without improvement after
            which training stops. Of type int.
    :return: List of enabled callbacks.
    """
    callbacks_list = []
    timestamp = datetime.now().strftime("%d-%m-%y_%H_%M_%S")

    if cb_params['tensorboard']:
        callbacks_list.append(callbacks.TensorBoard(
            log_dir='log/fits/fit_' + timestamp + '_' + name
        ))

    if cb_params['modelcheckpoint']:
        callbacks_list.append(callbacks.ModelCheckpoint(
            'data/models/model_' + timestamp + '_' + name,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ))

    if cb_params['earlystopping']:
        callbacks_list.append(callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=cb_params['stopping_delta'],
            patience=cb_params['stopping_patience'],
            verbose=1
        ))

    return callbacks_list


def _make_params() -> dict:
    """Make training parameters dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--name',
        nargs='?',
        type=str,
        help='Name of the experiment run, will be used to save models and\
              logs. Overrides name form params.yaml, may be suitable to make\
              a run independent of dvc.'
    )

    args = parser.parse_args()
    with open("params.yaml", 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)

    if args.name is not None:
        params['name'] = args.name

    return params


def main(params):
    """Run training with given params."""
    load_params = params['load']
    train_ds, val_ds = load_train_dataset(
        directory=Path('data/train'),
        batch_size=load_params['batch_size'],
        img_size=load_params['img_size'],
        validation_split=load_params['validation_split'],
    )
    img_shape = (load_params['img_size'], load_params['img_size'], 3)
    model = make_regularized_cnn(params['name'], input_shape=img_shape)
    train(train_ds, val_ds, model, params['train'])


if __name__ == '__main__':
    PARAMS = _make_params()
    main(PARAMS)
