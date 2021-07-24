"""
This script was obtained from imbalanced-learn's documentation site found
here: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html#imblearn.over_sampling.SMOTE .

The code is made available by the authors under the MIT license.

Copyright (c) 2014-2020 The imbalanced-learn developers.
All rights reserved
SPDX-License-Identifier: MIT
"""
import argparse

from imblearn import FunctionSampler
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from pyloras import LORAS

sns.set_context("poster")


def create_dataset(
    n_samples=1000,
    weights=(0.01, 0.01, 0.98),
    n_classes=3,
    class_sep=0.8,
    n_clusters=1,
):
    return make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters,
        weights=list(weights),
        class_sep=class_sep,
        random_state=0,
    )


def plot_resampling(X, y, sampler, ax, title=None):
    X_res, y_res = sampler.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor="k")
    if title is None:
        title = f"Resampling with {sampler.__class__.__name__}"
    ax.set_title(title)
    sns.despine(ax=ax, offset=10)


def plot_decision_function(X, y, clf, ax, title=None):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor="k")
    if title is not None:
        ax.set_title(title)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_neighbors', type=int, default=None)
    parser.add_argument('--n_shadow', type=int, default=None)
    parser.add_argument('--n_affine', type=int, default=None)

    args = parser.parse_args()
    X, y = create_dataset(n_samples=150, weights=(0.1, 0.2, 0.7))
    loras_params = dict(
        random_state=0,
        n_affine=args.n_affine,
        n_neighbors=args.n_neighbors,
    )
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

    samplers = [
        FunctionSampler(),
        SMOTE(random_state=0),
        LORAS(**loras_params),
    ]

    for ax, sampler in zip(axs.ravel(), samplers):
        title = "Original dataset" if isinstance(sampler, FunctionSampler) else None
        plot_resampling(X, y, sampler, ax, title=title)
    fig.tight_layout()
    plt.savefig('./scripts/img/resampled_data.svg')

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    clf = LogisticRegression()

    models = {
        "Without sampler": clf,
        "SMOTE sampler": make_pipeline(SMOTE(random_state=0), clf),
        "LORAS sampler": make_pipeline(LORAS(**loras_params), clf),
    }

    for ax, (title, model) in zip(axs, models.items()):
        model.fit(X, y)
        plot_decision_function(X, y, model, ax=ax, title=title)

    fig.suptitle(f"Decision function using a {clf.__class__.__name__}")
    fig.tight_layout()
    plt.savefig('./scripts/img/decision_fn.svg')

    X, y = create_dataset(n_samples=5000, weights=(0.01, 0.05, 0.94), class_sep=0.8)
    samplers = [SMOTE(random_state=0), LORAS(**loras_params)]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    for ax, sampler in zip(axs, samplers):
        model = make_pipeline(sampler, clf).fit(X, y)
        plot_decision_function(
            X, y, clf, ax[0], title=f"Decision function with {sampler.__class__.__name__}"
        )
        plot_resampling(X, y, sampler, ax[1])

    fig.suptitle("Particularities of over-sampling with SMOTE and LORAS")
    fig.tight_layout()
    plt.savefig('./scripts/img/particularities.svg')
