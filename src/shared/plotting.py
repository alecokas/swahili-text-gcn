import matplotlib.pyplot as plt
import pandas as pd


def process_resutls_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_dict = {
        'GCN text2vec': 'Text GCN-t2v',
        'GCN one-hot': 'Text GCN',
        'TF-IDF + LR': 'TF-IDF',
        'fastText + LR': 'fastText',
        'Counts + LR': 'Counts',
        'doc2vec DBOW': 'PV-DBOW',
        'doc2vec DM': 'PV-DM',
    }
    df.model = df.model.map(rename_dict)
    return df


def show_training_prop_stats(df: pd.DataFrame) -> pd.DataFrame:
    gcn_models = ['Text GCN', 'Text GCN-t2v']
    exp_filter = (df.model.isin(gcn_models)) & (df.window_size != 30)
    df_exp = df[~exp_filter].copy()

    stats = ['mean', 'std']
    agg = {'acc': stats, 'f1': stats}

    training_prop_stats = df_exp.groupby(['model', 'train_prop']).agg(agg)

    return training_prop_stats


def show_window_size_stats(df: pd.DataFrame) -> pd.DataFrame:
    gcn_one_hot_model = ['Text GCN']
    exp_filter = (df.model.isin(gcn_one_hot_model)) & (df.train_prop == 20)
    df_exp = df[exp_filter].copy()

    stats = ['mean', 'std']
    agg = {'acc': stats, 'f1': stats}

    window_size_stats = df_exp.groupby(['model', 'window_size']).agg(agg)

    return window_size_stats


def plot_training_prop(df: pd.DataFrame, plot_dir: str = '', metric: str = 'f1') -> None:
    training_prop_stats = show_training_prop_stats(df)

    plt.rc('font', family='Times New Roman')
    metric2name = {'acc': 'Accuracy', 'f1': '$F_{1}$ Score (%)'}
    colors = 'blue green red orange purple'.split()
    markers = 'o ^ s D p'.split()
    models = ['Text GCN', 'Text GCN-t2v', 'TF-IDF', 'Counts', 'PV-DBOW']

    for color, marker, model in zip(colors, markers, models):
        y = training_prop_stats[metric]['mean'].loc[model]
        std = training_prop_stats[metric]['std'].loc[model]
        x = training_prop_stats.loc[model].index
        _ = plt.errorbar(x, y, std, c=color, marker=marker, label=model, capsize=5)

    ax, fig = plt.gca(), plt.gcf()

    ax.set_ylabel(metric2name[metric], size=15)
    ax.set_xlabel("Training set labelled proportion (%)", size=15)
    ax.grid(which="both")
    plt.legend(fontsize=10)

    fig.set_size_inches(6, 4)
    plt.tight_layout()
    save_path = f'{plot_dir}/training_prop_{metric}.pdf'
    plt.savefig(save_path, format='pdf')


def plot_window_size(df: pd.DataFrame, plot_dir: str = '', metric: str = 'f1') -> None:
    window_size_stats = show_window_size_stats(df)

    plt.rc('font', family='Times New Roman')
    metric2name = {'acc': 'Accuracy', 'f1': '$F_{1}$ Score (%)'}
    colors = ['blue']
    markers = ['o']
    models = ['Text GCN']
    metric = 'f1'
    fontname = 'Times New Roman'

    for color, marker, model in zip(colors, markers, models):
        y = window_size_stats[metric]['mean'].loc[model]
        std = window_size_stats[metric]['std'].loc[model]
        x = window_size_stats.loc[model].index
        _ = plt.errorbar(x, y, std, c=color, marker=marker, label=model, fmt='o', capsize=5)

    ax, fig = plt.gca(), plt.gcf()

    ax.set_ylabel(metric2name[metric], size=15)
    ax.set_xlabel("Window Size", size=15)
    ax.grid(which="both")
    # ax.legend(fontsize=10)

    fig.set_size_inches(6, 4)
    plt.tight_layout()
    save_path = f'{plot_dir}/window_size_{metric}.pdf'
    plt.savefig(save_path, format='pdf')
