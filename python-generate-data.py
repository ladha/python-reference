import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ----- make_regression -----
from sklearn.datasets.samples_generator import make_regression

X, y = make_regression(n_samples=200, n_features=1, noise=0.5, random_state=101)
df = pd.DataFrame(dict(x=X[:, 0], y=y))

sns.lmplot('x', 'y', data=df, fit_reg=False)
plt.show()


# ----- make_blobs -----
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=200, centers=4, n_features=2, random_state=101)
df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))

sns.lmplot('x', 'y', data=df, hue='label', fit_reg=False)
plt.show()


# ----- make_circles -----
from sklearn.datasets.samples_generator import make_circles

X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=101)
df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))

sns.lmplot('x', 'y', data=df, hue='label', fit_reg=False)
plt.show()


# ----- make_moons -----
from sklearn.datasets.samples_generator import make_moons

X, y = make_moons(n_samples=200, noise=0.1, random_state=101)
df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))

sns.lmplot('x', 'y', data=df, hue='label', fit_reg=False)
plt.show()
