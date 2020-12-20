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


def evaluate(test_ds: Dataset, model: Model, test_params: dict):
    """Evaluate model on given data.

    :param test_ds: Test dataset.
    :param model: Binary classifier model to test.
    :param test_params: Dict with test params. Must include keys:
        threshold: Binary classification threshold of type float.
    """
    threshold = test_params['threshold']

    eval_results = model.evaluate(test_ds)

    y_pred = model.predict(test_ds)
    y_true = datagen_to_labels_array(test_ds)

    y_pred = y_pred.ravel()
    y_pred_classes = y_pred > threshold
    y_true = y_true.ravel()

    class_report = classification_report(y_true, y_pred_classes)
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    single_precision_score = precision_score(y_true, y_pred_classes)
    single_recall_score = recall_score(y_true, y_pred_classes)
    single_f1_score = f1_score(y_true, y_pred_classes)

    pr, re, pr_thr = precision_recall_curve(y_true, y_pred)
    fpr, tpr, roc_thr = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)

    print('\nModel metrics evaluation:\nTest loss, test acc:', eval_results)
    print('\nAUC score:', auc_score)
    print('\nClassfication report:\n', class_report)
    print('\nConfusion matrix:\n', conf_matrix)

    with open('log/scores.json', 'w') as fd:
        json.dump([
            {'loss': eval_results[0]},
            {'accuracy': eval_results[1]},
            {'TP': str(conf_matrix[0][0])},
            {'FP': str(conf_matrix[0][1])},
            {'FN': str(conf_matrix[1][0])},
            {'TN': str(conf_matrix[1][1])},
            {'precision': str(single_precision_score)},
            {'recall': str(single_recall_score)},
            {'f1_score': str(single_f1_score)},
            {'auc': str(auc_score)}],
            fd
        )

    with open('log/plots.json', 'w') as fd:
        json.dump([
            {'activation_hist': [
                {
                    'activation': str(a),
                } for a in y_pred]},
            {'precision_recall': [
                {
                    'precision': str(p),
                    'recall': str(r),
                    'threshold': str(t)
                } for p, r, t in zip(pr, re, pr_thr)]},
            {'roc': [
                {
                    'fpr': str(fp),
                    'tpr': str(tp),
                    'threshold': str(th)
                } for fp, tp, th in zip(fpr, tpr, roc_thr)]}
            ],
            fd
        )

    plot_activation_hist(y_pred, Path('log'))
    plot_precision_recall(pr, re, pr_thr, Path('log'))
    plot_roc(fpr, tpr, roc_thr, auc_score, Path('log'))


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


def plot_activation_hist(y_pred: np.ndarray, output_path: Path):
    """Plot output activations histogram."""
    fig = px.histogram(y_pred)
    fig.write_html(str(output_path/'activation_hist.html'))


def plot_precision_recall(prec: np.ndarray,
                          rec: np.ndarray,
                          thr: float,
                          output_dir: Path):
    """Plot and save precision/recall curve in given directory.

    :param prec: Precision scores.
    :param rec: Recall scores.
    :param thre: Thresholds.
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
    fig.write_html(str(output_dir/'prec_recall.html'))


def plot_roc(fpr: np.ndarray,
             tpr: np.ndarray,
             thr: float,
             auc_score: float,
             output_dir: Path):
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
    fig.write_html(str(output_dir/'roc.html'))


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
    evaluate(test_ds, model, params['test'])


if __name__ == '__main__':
    PARAMS = make_params()
    main(PARAMS)
