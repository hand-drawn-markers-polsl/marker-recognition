from pathlib import Path

import numpy as np
from plotly import express as px
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix

from make_models import load_model
from dataset_loading import load_test_dataset


def evaluate(test_ds, model):
    threshold = 0.5

    results = model.evaluate(test_ds)
    print('\nModel metrics evaluation:\ntest loss, test acc:', results)

    y_pred = model.predict(test_ds)
    y_true = datagen_to_labels_array(test_ds)

    y_pred = y_pred.ravel()
    y_pred_classes = y_pred > threshold
    y_true = y_true.ravel()

    results = classification_report(y_true, y_pred_classes)
    print('\nClassfication report:\n', results)

    results = confusion_matrix(y_true, y_pred_classes)
    print('\nConfusion matrix::\n', results)

    save_activation_hist(y_pred, Path('log'))
    save_precission_recall_plot(y_true, y_pred, Path('log'))
    save_roc_plot(y_true, y_pred, Path('log'))


def save_activation_hist(y_pred, output_path):
    fig = px.histogram(y_pred)
    fig.write_html(str(output_path / 'activation_hist.html'))


def save_precission_recall_plot(y_true, y_pred, output_dir):
    fpr, fre, thr = precision_recall_curve(y_true, y_pred)

    fig = px.area(
        x=fre, y=fpr, hover_data={'treshold': np.insert(thr, 0, 1)},
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


def save_roc_plot(y_true, y_pred, output_dir):
    fpr, tpr, thr = roc_curve(y_true, y_pred)

    fig = px.area(
        x=fpr, y=tpr, hover_data={'treshold': thr},
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.write_html(str(output_dir/'roc.html'))


def datagen_to_labels_array(datagen):
    labels_idx = 1
    true_class_idx = 1
    ret = []
    # Keras internals force usage of range(len()) here
    for batch_iter in range(len(datagen)):
        ret.append(datagen[batch_iter][labels_idx].transpose()[true_class_idx])

    return np.concatenate(ret)


def main():
    model = load_model('simple_regularized_cnn')
    test_ds = load_test_dataset(Path('data/test'))
    evaluate(test_ds, model)


if __name__ == '__main__':
    main()
