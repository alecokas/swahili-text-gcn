import os
import pandas as pd
import matplotlib.pyplot as plt

from shared.utils import rm_file, read_json_as_dict, append_to_jsonl, read_jsonl


def remove_previous_best_model(directory: str, current_step: int) -> None:
    for item in os.listdir(directory):
        if f'model-{current_step}' not in item:
            rm_file(os.path.join(directory, item))


def remove_previous_best_predictions(directory: str, current_step: int) -> None:
    for item in os.listdir(directory):
        if f'predictions-{current_step}' not in item:
            rm_file(os.path.join(directory, item))


def get_vocab_size(graph_dir: str) -> int:
    return len(read_json_as_dict(os.path.join(graph_dir, 'vocab_map.json')))


def save_training_notes(file_path: str, epoch_num: int, note: str):
    training_notes = {'epoch': epoch_num}
    training_notes['note'] = note
    append_to_jsonl(file_path, training_notes)


def load_training_log_to_df(training_log_path):
    df = pd.DataFrame(read_jsonl(training_log_path))
    df.columns = [c.replace(" ", "_").lower() for c in df.columns]
    return df


def create_training_plot(training_history, name="training_history"):
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(training_history.epoch, training_history.accuracy, c="blue")
    axes[0].set_ylabel("Accuracy", size=20)
    axes[0].grid(which="both")

    axes[1].plot(training_history.epoch, training_history.val_loss, c="green", label='Validation')
    axes[1].plot(training_history.epoch, training_history.train_loss, c="red", label='Train')
    axes[1].set_ylabel("Loss", size=20)
    axes[1].set_xlabel("Epoch", size=20)
    axes[1].grid(which="both")
    axes[1].legend(fontsize=15)

    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    plt.tight_layout()
    plt.savefig(f"{name}.jpg", dpi=200)
