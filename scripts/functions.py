# general and plots
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

# model selection
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

# imbalanced
from imblearn.pipeline import Pipeline

# metrics
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import auc
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# some warning may arise during GridSearchCV, and they can be ignored
# comment if testing other parameters
import os
os.environ["PYTHONWARNINGS"] = "ignore"

# constants
RANDOM_STATE = 42
JOBS = -2
RSKF_SPLITS = 5
RSKF_REPEATS = 3
CONF_MATRIX_COLORMAP = 'RdBu_r'


def pr_auc(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)


scoring = {
    'Accuracy': make_scorer(accuracy_score),
    'Precision': make_scorer(precision_score),
    'Recall': make_scorer(recall_score),
    'F1 score': make_scorer(f1_score),
    'F2 score': make_scorer(fbeta_score, beta=2),
    'AUROC': make_scorer(roc_auc_score),
    # 'auprc': make_scorer(pr_auc, needs_proba=True),  # interpolation method
    'AUPRC': make_scorer(average_precision_score),
}


def evaluate_model(X, y, steps, rskf_splits=RSKF_SPLITS,
                   rskf_repeats=RSKF_REPEATS, jobs=JOBS):
    rskf = RepeatedStratifiedKFold(n_splits=rskf_splits,
                                   n_repeats=rskf_repeats,
                                   random_state=RANDOM_STATE)
    pipeline = Pipeline(steps)
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=rskf,
                            n_jobs=jobs)
    return scores


def mean_confusion_matrices(X, y, steps, rskf_splits=RSKF_SPLITS,
                            rskf_repeats=RSKF_REPEATS,
                            normalize=None):
    rsk = RepeatedStratifiedKFold(n_splits=rskf_splits, n_repeats=rskf_repeats,
                                  random_state=RANDOM_STATE)

    pipeline = Pipeline(steps)

    cm_arrays = []

    for train_index, test_index in rsk.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        pipeline.fit(X_train, y_train)
        cm = confusion_matrix(y_test, pipeline.predict(X_test),
                              normalize=normalize)
        cm_arrays.append(cm)

    return np.mean(cm_arrays, axis=0)


def grid_search(X, y, steps, params, scoring='roc_auc',
                rskf_splits=RSKF_SPLITS, rskf_repeats=RSKF_REPEATS,
                verb=5, jobs=JOBS):

    rskf = RepeatedStratifiedKFold(n_splits=rskf_splits,
                                   n_repeats=rskf_repeats,
                                   random_state=RANDOM_STATE)

    pipeline = Pipeline(steps)

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        pipeline.fit(X_train, y_train)
        search = GridSearchCV(pipeline, params, scoring=scoring,
                              n_jobs=jobs, verbose=verb)
        search.fit(X_train, y_train)

    return search.best_estimator_


def score_table(scores):
    print('{0:^15} | {1:^10} | {2:^10}'.format('Metric', 'Mean', 'Std Dev'))
    print('-' * 40)
    for key in scoring.keys():
        metric = ''.join(('test_', key))
        mean = np.mean(scores[metric])
        std_dev = np.std(scores[metric])
        print(
            '{0:<15} | {1:^10.3f} | {2:^10.3f}'.format(key, mean, std_dev))


def model_summary(X, y, models, steps_before_model, jobs=JOBS):
    results = []
    names = []

    for name, model in models:
        names.append(name)
        steps = steps_before_model.copy()
        steps.append(('model', model))
        scores = evaluate_model(X, y, steps, jobs=jobs)
        results.append(scores)

    for name, result in zip(names, results):
        print(name)
        score_table(result)
        print()

    return results


def hyperparameter_tuning(X, y, models_with_params, steps_before_model,
                          scoring='roc_auc',
                          rskf_splits=RSKF_SPLITS, rskf_repeats=RSKF_REPEATS,
                          verb=5, jobs=JOBS):
    results = []
    names = []
    models_params = deepcopy(models_with_params)

    for name, model, params in models_params:
        print(f'\nBeginning for model {name}...')
        names.append(name)

        new_param_keys = [('model' + '__' + key) for key in params.keys()]
        for key, new_key in zip(params.copy().keys(), new_param_keys):
            params[new_key] = params.pop(key)
        steps = steps_before_model.copy()
        steps.append(('model', model))
        scores = grid_search(X, y, steps, params,
                             scoring=scoring,
                             rskf_splits=rskf_splits,
                             rskf_repeats=rskf_repeats,
                             verb=verb,
                             jobs=jobs)
        results.append(scores)

    print('End')
    return results


def confusion_matrix_norm(X, y, models, steps_before_model,
                          normalize='true'):
    names = []
    cm_means_norm = []

    for name, model in models:
        names.append(name)
        steps = steps_before_model.copy()
        steps.append(('model', model))
        cm_means_norm.append(
            mean_confusion_matrices(X, y, steps, normalize=normalize))

    return cm_means_norm


def confusion_matrix_plot(models, conf_matrices, nrows, ncols, figsize,
                          remove_empty_axes=0):
    names = [n for n, _ in models]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for name, matrix, ax in zip(names, conf_matrices, axes.flatten()):
        disp = ConfusionMatrixDisplay(matrix)
        disp.plot(ax=ax, cmap=CONF_MATRIX_COLORMAP, colorbar=True,
                  im_kw={'vmin': 0, 'vmax': 1})
        ax.grid(False)
        ax.set_title(name)

    if remove_empty_axes:
        for ax in axes.flatten()[-remove_empty_axes:]:
            ax.set_visible(False)

    plt.show()
