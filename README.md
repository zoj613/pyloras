# LoRAS
Localized Random Affine Shadowsampling

This repo provides a python implementation of an imbalanced dataset oversampling
technique known as Localized Random Affine Shadowsampling (LoRAS). This implementation 
piggybacks off the package ``imbalanced-learn`` and thus aims to be as compatible
as possible with it.


## Dependencies
- `imbalanced-learn`


## Installation

Using `pip`:
```shell
$ pip install -U pyloras
```

Installing from source requires an installation of [poetry][1] and the following shell commands:
```shell
$ git clone https://github.com/zoj613/pyloras.git
$ cd pyloras/
$ poetry install
# add package to python's path
$ export PYTHONPATH=$PWD:$PYTHONPATH 
```

## Usage

```python
from collections import Counter
from pyloras import LORAS
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=20000, n_features=5, n_informative=5,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=0)

lrs = LORAS(random_state=0, embedding_params={'perplexity': 35, 'n_iter': 250})
print(sorted(Counter(y).items()))
# [(0, 270), (1, 1056), (2, 18674)]
X_resampled, y_resampled = lrs.fit_resample(X, y)
print(sorted(Counter(y_resampled.astype(int)).items()))
# [(0, 18674), (1, 18674), (2, 18674)]
```

## References
Bej, S., Davtyan, N., Wolfien, M. et al. LoRAS: an oversampling approach for imbalanced datasets. Mach Learn 110, 279–301 (2021). https://doi.org/10.1007/s10994-020-05913-4


[1]: https://python-poetry.org/docs/pyproject/
