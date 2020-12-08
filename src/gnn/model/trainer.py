import math
import os
import time
from tqdm import trange
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Any, Dict

from gnn.utils.metrics import accuracy, save_metrics
from gnn.utils.utils import remove_previous_best
from shared.utils import mkdir


class Trainer(object):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        device: torch.device,
        train_nodes: torch.LongTensor,
        val_nodes: torch.LongTensor,
        results_dir: str,
        validate_every_n_epochs: int,
        save_after_n_epochs: int,
        checkpoint_every_n_epochs: int,
    ):
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.optimiser = AdamW(
            params=model.parameters(),
            lr=learning_rate,
        )
        self.loss_fn = nn.CrossEntropyLoss()

        assert (
            len(set(train_nodes).intersection(set(val_nodes))) == 0
        ), f'There are overlapping nodes: {len(set(train_nodes).intersection(set(val_nodes)))}'
        self.train_nodes = train_nodes
        self.val_nodes = val_nodes

        self.results_dir = results_dir
        self.validate_every_n_epochs = validate_every_n_epochs
        self.save_after_n_epochs = save_after_n_epochs
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.has_saved_metric = False
        self._setup_dirs()

        self.metric_of_interest = 'val loss'
        self.best_metric = math.inf

    def _setup_dirs(self):
        self.ckpt_dir = os.path.join(self.results_dir, 'ckpt')
        self.best_model_dir = os.path.join(self.results_dir, 'best')
        mkdir(self.ckpt_dir)
        mkdir(self.best_model_dir)

    def __call__(
        self,
        input_features: torch.FloatTensor,
        adjacency: torch.sparse.FloatTensor,
        labels: torch.LongTensor,
        num_epochs: int,
    ):
        with trange(num_epochs, desc='Training progress: ') as t:
            for epoch_num in range(1, num_epochs + 1):
                train_metrics = self._train_epoch(input_features, adjacency, labels)

                if (epoch_num % self.validate_every_n_epochs) == 0 or epoch_num == 1:
                    # Validate and save metrics
                    val_metrics = self._val_epoch(input_features, adjacency, labels)
                    save_metrics(
                        file_path=os.path.join(self.results_dir, 'train-log.jsonl'),
                        epoch_num=epoch_num,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        is_first_metric_save=not self.has_saved_metric,
                    )
                    self.has_saved_metric = True

                    if epoch_num > self.save_after_n_epochs and (epoch_num % self.checkpoint_every_n_epochs) == 0:
                        # Save model
                        self._checkpoint_model(epoch_num)
                        if self._is_best(val_metrics):
                            self._save_best_model(epoch_num)
                else:
                    # if we haven't validated, create an empt val metric dict
                    val_metrics = {'val loss': None}

                t.set_postfix(
                    train_loss=train_metrics['train loss'],
                    val_loss=val_metrics['val loss']
                )
                t.update()

        return None

    def _train_epoch(
        self,
        input_features: torch.FloatTensor,
        adjacency: torch.sparse.FloatTensor,
        labels: torch.LongTensor,
    ) -> Dict[str, Any]:
        """
        NOTE: Although we pass in all input features and labels,
              we only evaluate the loss on training set node indicies.
        """
        start_time = time.time()

        self.model.train()
        self.optimiser.zero_grad()
        logits = self.model(input_features, adjacency)
        train_loss = self.loss_fn(logits[self.train_nodes], labels[self.train_nodes])

        train_loss.backward()
        self.optimiser.step()

        # print(f'train loss: {train_loss}')

        duration = time.time() - start_time
        return {'train epoch duration': duration, 'train loss': train_loss.item()}

    def _val_epoch(
        self,
        input_features: torch.FloatTensor,
        adjacency: torch.sparse.FloatTensor,
        labels: torch.LongTensor,
    ) -> Dict[str, Any]:
        self.model.eval()
        logits = self.model(input_features, adjacency)
        val_loss = self.loss_fn(logits[self.val_nodes], labels[self.val_nodes])

        # print(f'val loss: {val_loss}')

        val_accuracy = accuracy(logits[self.val_nodes], labels[self.val_nodes], is_logit_output=True)
        return {'val loss': val_loss.item(), 'F-score': None, 'Accuracy': val_accuracy}

    def _checkpoint_model(self, epoch: int) -> None:
        """ Checkpoint to resume training """
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimiser_state_dict': self.optimiser.state_dict(),
                'loss': self.loss_fn,
            },
            os.path.join(self.ckpt_dir, f'model-{epoch}.pt'),
        )

    def _save_best_model(self, epoch: int) -> None:
        """ Save best model for inference """
        torch.save(self.model.state_dict(), os.path.join(self.best_model_dir, f'model-{epoch}.pt'))
        remove_previous_best(self.best_model_dir, epoch)

    def _is_best(self, val_metrics: Dict[str, float]) -> bool:
        if 'loss' in self.metric_of_interest:
            if val_metrics[self.metric_of_interest] < self.best_metric:
                self.best_metric = val_metrics[self.metric_of_interest]
                return True
            else:
                return False
        else:
            # Assume we want to maximise it if it is not a loss
            if val_metrics[self.metric_of_interest] > self.best_metric:
                self.best_metric = val_metrics[self.metric_of_interest]
                return True
            else:
                return False
