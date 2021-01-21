import numpy as np
import os
from sklearn.manifold import TSNE
import seaborn as sns


def tsne_plot(input_x: np.ndarray, labels_y: np.ndarray, plot_dir: str, num_samples: int = 8000) -> None:
    """ t-sne reduce and then save image to plot_dir """
    print('t-sne')
    reduced_x = TSNE(n_components=2, n_iter=500, learning_rate=200, perplexity=200, verbose=1).fit_transform(
        input_x[:num_samples, :]
    )
    print('Plotting...')
    sns_plot = sns.scatterplot(x=reduced_x[:, 0], y=reduced_x[:, 1], hue=labels_y[:num_samples])
    sns_plot.set(xlim=(-50, 50))
    sns_plot.set(ylim=(-50, 50))
    print('Saving...')
    fig = sns_plot.get_figure()
    fig.savefig(os.path.join(plot_dir, "output.png"))
