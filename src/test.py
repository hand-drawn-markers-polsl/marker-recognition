"""Test/evaluate classification models."""

from pathlib import Path
import argparse
import json
import yaml

from plotly import express as px
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.data import Dataset
from tensorflow.keras.models import Model
import numpy as np

from dataset_loading import load_test_dataset
from make_models import load_model


def evaluate(model: Model, test_ds: Dataset, test_params: dict):
    """Evaluate model on given data.

    :param test_ds: Test dataset.
    :param model: Binary classifier model to test.
    :param test_params: Dict with test params. Must include keys:
        threshold: Binary classification threshold of type float.
    """
    y_true, y_pred, y_pred_classes = make_eval_preds(
        model,
        test_ds,
        test_params['threshold']
    )

    m = make_metrics(y_true, y_pred, y_pred_classes)
    m['eval_results'] = model.evaluate(test_ds)

    print('\nModel metrics (Keras default threshold) evaluation:'
          '\nTest loss, test acc:', m['eval_results'])
    print('\nAUC score:', m['roc']['auc_score'])
    print('\nClassfication report:\n', m['class_report'])
    print('\nConfusion matrix:\n', m['conf_matrix'])

    plot_activation_hist(y_pred)
    plot_precision_recall(**m['prec_rec_curve'])
    plot_roc(**m['roc'])

    save_metrics(m)


def make_eval_preds(model: Model, test_ds: Dataset, threshold=0.5) -> tuple:
    """Make evaluation predictions for given model and data.

    :param model: Binary classifier model to test.
    :param test_ds: Test dataset.
    :param threshold: Binary classification threshold used for class
        predictions.
    :return: List containing labels: [y_true, y_pred, y_pred_classes].
    """
    y_pred = model.predict(test_ds)
    y_true = datagen_to_labels_array(test_ds)

    y_pred = y_pred.ravel()
    y_pred_classes = y_pred > threshold
    y_true = y_true.ravel()

    return y_true, y_pred, y_pred_classes


def make_metrics(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 y_pred_classes: np.ndarray) -> dict:
    """Make evaluation metrics, based on predictions.

    :param y_true: Ground truth labels.
    :param y_pred: Classification predictions (probabilities with values
        from 0 to 1).
    :param y_pred_classes: Classification class predictions (predicted
        labels - either 0 or 1).
    :return: Dict with metrics. There are many of them, it is best to examine
        their keys in the body of this function.
    """
    metrics = {}
    m = metrics
    m['class_report'] = classification_report(y_true, y_pred_classes)
    m['conf_matrix'] = confusion_matrix(y_true, y_pred_classes)
    m['precision'] = precision_score(y_true, y_pred_classes)
    m['recall'] = recall_score(y_true, y_pred_classes)
    m['f1_score'] = f1_score(y_true, y_pred_classes)

    m['prec_rec_curve'] = {}
    prec, rec, pr_thr = precision_recall_curve(y_true, y_pred)
    m['prec_rec_curve']['prec'] = prec
    m['prec_rec_curve']['rec'] = rec
    m['prec_rec_curve']['thr'] = pr_thr

    m['roc'] = {}
    fpr, tpr, roc_thr = roc_curve(y_true, y_pred)
    m['roc']['fpr'] = fpr
    m['roc']['tpr'] = tpr
    m['roc']['thr'] = roc_thr
    m['roc']['auc_score'] = auc(fpr, tpr)

    return metrics


def save_metrics(metrics: dict, path=Path('data/metrics.json')):
    """Save classification metrics in json file.

    To see what keys should be included in 'metrics' dict see the body
    of this function.
    """
    m = metrics
    with open(path, 'w') as file_desc:
        json.dump(
            {
                'test': {
                    'loss': m['eval_results'][0],
                    'accuracy': m['eval_results'][1],
                    'TP': int(m['conf_matrix'][0][0]),
                    'FP': int(m['conf_matrix'][0][1]),
                    'FN': int(m['conf_matrix'][1][0]),
                    'TN': int(m['conf_matrix'][1][1]),
                    'precision': m['precision'],
                    'recall': m['recall'],
                    'f1_score': m['f1_score'],
                    'auc': m['roc']['auc_score']
                }
            },
            file_desc
        )


def datagen_to_labels_array(datagen: Dataset) -> np.ndarray:
    """Iterate data generator batches and return labels array.

    :param datagen: Tensorflow data generator, which iterates like:
        [batches][x/y(labels)].
    :return: Array of y values (labels) from generator.
    """
    labels_idx = 1
    true_class_idx = 1
    ret = []
    # Keras internals force usage of range(len()) here
    for batch_iter in range(len(datagen)):
        ret.append(datagen[batch_iter][labels_idx].transpose()[true_class_idx])

    return np.concatenate(ret)


def plot_activation_hist(y_pred: np.ndarray, output_dir=Path('log')):
    """Plot output activations histogram."""
    fig = px.histogram(y_pred)

    output_file = str(output_dir/'activation_hist.html')
    fig.write_html(output_file)
    print(f'Saved activations histogram to {output_file}.')


def plot_precision_recall(prec: np.ndarray,
                          rec: np.ndarray,
                          thr: float,
                          output_dir=Path('log')):
    """Plot and save precision/recall curve in given directory.

    :param prec: Precision scores.
    :param rec: Recall scores.
    :param thr: Thresholds.
    """
    fig = px.area(
        x=rec, y=prec, hover_data={'treshold': np.insert(thr, 0, 1)},
        title='Precision-Recall Curve',
        labels=dict(x='Recall', y='Precision'),
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    fig.update_xaxes(constrain='domain')

    output_file = str(output_dir/'prec_recall.html')
    fig.write_html(output_file)
    print(f'Saved precision/recall curve to {output_file}.')


def plot_roc(fpr: np.ndarray,
             tpr: np.ndarray,
             thr: float,
             auc_score: float,
             output_dir=Path('log')):
    """Plot and save roc curve in given directory.

    :param fpr: False positives rates.
    :param tpr: True positives rates.
    :param thr: Thresholds.
    Other params should be obvious.
    """
    fig = px.area(
        x=fpr, y=tpr, hover_data={'treshold': thr},
        title=f'ROC Curve (AUC={auc_score:.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    fig.update_xaxes(constrain='domain')

    output_file = str(output_dir/'roc.html')
    fig.write_html(output_file)
    print(f'Saved roc curve in {output_file}.')


def make_params() -> dict:
    """Make training parameters dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--name',
        nargs='?',
        type=str,
        help='Name of the model to test. Overrides name form params.yaml,\
              may be suitable to make a run independent of dvc.'
    )

    args = parser.parse_args()
    with open("params.yaml", 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)

    if args.name is not None:
        params['name'] = args.name

    return params


def main(params):
    """Run evaluation with given params."""
    model = load_model(params['name'])
    test_ds = load_test_dataset(Path('data/test'))
    evaluate(model, test_ds, params['test'])


if __name__ == '__main__':
    PARAMS = make_params()
    main(PARAMS)
