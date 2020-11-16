from datetime import datetime

from tensorflow.keras import callbacks, optimizers

from load_dataset import load_partial_dataset


def make_callbacks(params):
    callbacks_list = []

    timestamp = datetime.now().strftime("%y-%m-%d-%H:%M:%S")

    if params['tensorboard']:
        callbacks_list.append(callbacks.TensorBoard(
            log_dir='log/fits/fit_' + timestamp + '_' + params['name']
        ))

    if params['modelcheckpoint']:
        callbacks_list.append(callbacks.ModelCheckpoint(
            'log/models/model_' + timestamp + '_' + params['name'],
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ))

    if params['earlystopping']:
        callbacks_list.append(callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0.001,
            patience=5,
            verbose=1
        ))

    return callbacks_list


def prepare_model(model, params):
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )


def fit_simple(model, params):
    train_ds, val_ds = load_partial_dataset(
        directory=params['img_directory'],
        batch_size=params['batch_size'],
        img_size=params['img_size'],
        validation_split=params['validation_split'],
    )

    prepare_model(model, params)
    callbacks_list = make_callbacks(params)

    model.fit(
        train_ds,
        epochs=params['epochs'],
        callbacks=callbacks_list,
        validation_data=val_ds
    )


def fit_cross_val(model, params):
    pass


def train(model, params):
    if params['cross_validation']:
        fit_cross_val(model, params)
    else:
        fit_simple(model, params)
