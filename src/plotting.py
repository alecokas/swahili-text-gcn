import matplotlib.pyplot as plt
import pandas as pd


def show_training_prop_stats(df: pd.DataFrame) -> pd.DataFrame:
    gcn_models = ['GCN one-hot', 'GCN text2vec']
    exp_filter = (df.model.isin(gcn_models)) & (df.window_size != 30)
    df_exp = df[~exp_filter].copy()

    stats = ['mean', 'std']
    agg = {'acc': stats, 'f1': stats}

    training_prop_stats = df_exp.groupby(['model', 'train_prop']).agg(agg)

    return training_prop_stats


def show_window_size_stats(df: pd.DataFrame) -> pd.DataFrame:
    gcn_one_hot_model = ['GCN one-hot']
    exp_filter = (df.model.isin(gcn_one_hot_model)) & (df.train_prop == 20)
    df_exp = df[exp_filter].copy()

    stats = ['mean', 'std']
    agg = {'acc': stats, 'f1': stats}

    window_size_stats = df_exp.groupby(['model', 'window_size']).agg(agg)

    return window_size_stats


def plot_training_prop(df: pd.DataFrame, plot_dir: str = '', metric: str = 'f1') -> None:
    training_prop_stats = show_training_prop_stats(df)

    metric2name = {'acc': 'Accuracy', 'f1': 'F-1 Score'}
    colors = 'blue green red orange'.split()
    models = ['GCN one-hot', 'GCN text2vec', 'TF-IDF + LR', 'doc2vec DBOW']

    for color, model in zip(colors, models):
        y = training_prop_stats[metric]['mean'].loc[model]
        std = training_prop_stats[metric]['std'].loc[model]
        x = training_prop_stats.loc[model].index
        _ = plt.errorbar(x, y, std, c=color, marker=None, label=model)

        ax, fig = plt.gca(), plt.gcf()

        ax.set_ylabel(metric2name[metric], size=15)
        ax.set_xlabel("Training set proportion", size=15)
        # ax.grid(which="both")
        ax.legend(fontsize=10)

        fig.set_size_inches(6, 4)
        plt.tight_layout()
        save_path = f'{plot_dir}/training_prop_{metric}'
        plt.savefig(save_path, dpi=200)


def plot_window_size(df: pd.DataFrame, plot_dir: str = '', metric: str = 'f1') -> None:
    window_size_stats = show_window_size_stats(df)

    metric2name = {'acc': 'Accuracy', 'f1': 'F-1 Score'}
    colors = ['blue']
    models = ['GCN one-hot']
    metric = 'f1'

    for color, model in zip(colors, models):
        y = window_size_stats[metric]['mean'].loc[model]
        std = window_size_stats[metric]['std'].loc[model]
        x = window_size_stats.loc[model].index
        _ = plt.errorbar(x, y, std, c=color, marker=None, label=model)

    ax, fig = plt.gca(), plt.gcf()

    ax.set_ylabel(metric2name[metric], size=15)
    ax.set_xlabel("Window Size", size=15)
    # ax.grid(which="both")
    # ax.legend(fontsize=10)

    fig.set_size_inches(6, 4)
    plt.tight_layout()
    save_path = f'{plot_dir}/window_size_{metric}'
    plt.savefig(save_path)
