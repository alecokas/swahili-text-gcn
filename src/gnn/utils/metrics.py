import torch
from torch.nn import functional as F
from typing import Dict

from shared.utils import append_to_jsonl, rm_file


def accuracy(output: torch.FloatTensor, labels: torch.LongTensor, is_logit_output: bool) -> float:
    if is_logit_output:
        output = F.softmax(output, dim=-1)
    _, predictions = torch.max(output, dim=-1)

    assert (
        predictions.shape == labels.shape
    ), f'Predictions and labels must have the same shape. Found {predictions.shape} != {labels.shape}'

    num_correct = torch.sum(torch.eq(predictions.type_as(labels), labels))
    return float(num_correct) / len(labels)


def save_metrics(
    file_path: str,
    epoch_num: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    is_first_metric_save: bool,
) -> None:
    if is_first_metric_save:
        rm_file(file_path)

    metrics = {'epoch': epoch_num}
    metrics.update(train_metrics)
    metrics.update(val_metrics)

    append_to_jsonl(file_path, metrics)
