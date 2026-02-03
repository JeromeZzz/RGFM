"""
Training utilities
Helper functions, classes and small tools used during model training.
"""

import json
import logging
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds to ensure reproducibility across Python, NumPy and PyTorch.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


class EarlyStopping:
    """
    Early-stopping helper.

    Monitors a score (e.g., validation loss/metric) and stops training when
    there is no sufficient improvement for a given patience.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = 'min',
        delta: float = 0.0001,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            patience: Number of epochs to wait without improvement
            mode: 'min' for minimizing score, 'max' for maximizing score
            delta: Minimum change to qualify as an improvement
            verbose: Whether to log state changes
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

        if mode == 'min':
            self.is_better = lambda a, b: a < b - delta
        else:
            self.is_better = lambda a, b: a > b + delta

    def __call__(self, score: float) -> bool:
        """
        Update internal state with the latest score and determine early stop.

        Args:
            score: Current validation score

        Returns:
            True if early stopping should be triggered, otherwise False
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return False

        self.counter += 1
        if self.verbose:
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                logger.info("Early stopping triggered")
            return True

        return False

    def reset(self) -> None:
        """Reset internal counters and state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    """
    Utility for saving and keeping the top-k model checkpoints based on a score.
    """

    def __init__(
        self,
        save_dir: str,
        save_top_k: int = 3,
        mode: str = 'min',
        verbose: bool = True,
    ) -> None:
        """
        Args:
            save_dir: Directory to store checkpoints
            save_top_k: Keep at most this many best checkpoints
            mode: 'min' (lower is better) or 'max' (higher is better)
            verbose: Whether to log actions
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_top_k = save_top_k
        self.mode = mode
        self.verbose = verbose

        # List of (score, path)
        self.checkpoints: List[tuple] = []

        if mode == 'min':
            self.is_better = lambda a, b: a < b
        else:
            self.is_better = lambda a, b: a > b

    def save(
        self,
        model: nn.Module,
        score: float,
        epoch: int,
        optimizer: Optional[Optimizer] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Save a model checkpoint.

        Args:
            model: Model to save
            score: Validation score used for ranking
            epoch: Epoch index
            optimizer: Optional optimizer to save
            metrics: Optional metrics to store
        """
        checkpoint: Dict[str, Any] = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score,
            'timestamp': datetime.now().isoformat(),
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if metrics:
            checkpoint['metrics'] = metrics

        # Filename and save
        filename = f"checkpoint_epoch{epoch}_score{score:.4f}.pt"
        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)

        if self.verbose:
            logger.info(f"Checkpoint saved to {filepath}")

        # Track and maintain top-k
        self.checkpoints.append((score, filepath))
        self._maintain_top_k()

    def _maintain_top_k(self) -> None:
        """Keep only the best top-k checkpoints on disk and in memory."""
        if len(self.checkpoints) <= self.save_top_k:
            return

        # Sort by score
        self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))

        # Delete extras from disk
        for score, path in self.checkpoints[self.save_top_k:]:
            if path.exists():
                path.unlink()
                if self.verbose:
                    logger.info(f"Deleted checkpoint {path}")

        # Keep top-k in memory
        self.checkpoints = self.checkpoints[:self.save_top_k]

    def load_best(self, model: nn.Module, optimizer: Optional[Optimizer] = None) -> Dict[str, Any]:
        """
        Load the best-scored checkpoint into the model (and optimizer if provided).

        Args:
            model: Model instance to load state_dict into
            optimizer: Optional optimizer to load state_dict into

        Returns:
            The loaded checkpoint dict
        """
        if not self.checkpoints:
            raise ValueError("No checkpoints available")

        # Pick best checkpoint
        self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))
        best_score, best_path = self.checkpoints[0]

        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.verbose:
            logger.info(f"Loaded best model from {best_path}, score: {best_score}")

        return checkpoint


class GradientAccumulator:
    """
    Gradient accumulation helper to simulate larger batch sizes.
    """

    def __init__(self, accumulation_steps: int = 1) -> None:
        """
        Args:
            accumulation_steps: Number of steps to accumulate before stepping
        """
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def should_step(self) -> bool:
        """Return True if an optimizer step should be taken now."""
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0

    def reset(self) -> None:
        """Reset internal counter."""
        self.step_count = 0


class MetricTracker:
    """
    Simple metric tracker that averages metrics over batches and epochs.
    """

    def __init__(self, metrics: List[str]) -> None:
        """
        Args:
            metrics: List of metric names to track
        """
        self.metrics = metrics
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.current_batch: Dict[str, List[float]] = defaultdict(list)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metrics for a single batch.

        Args:
            predictions: Model outputs/logits
            targets: Ground-truth targets
        """
        # Move to CPU for numpy operations
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()

        for metric in self.metrics:
            if metric == 'accuracy':
                if predictions.dim() > 1:
                    pred_classes = predictions.argmax(dim=1)
                else:
                    pred_classes = (predictions > 0.5).long()
                correct = (pred_classes == targets).float().mean().item()
                self.current_batch['accuracy'].append(correct)

            elif metric == 'f1':
                # Simplified binary F1 (for quick feedback)
                if predictions.dim() > 1:
                    pred_classes = predictions.argmax(dim=1)
                else:
                    pred_classes = (predictions > 0.5).long()

                tp = ((pred_classes == 1) & (targets == 1)).sum().float()
                fp = ((pred_classes == 1) & (targets == 0)).sum().float()
                fn = ((pred_classes == 0) & (targets == 1)).sum().float()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                self.current_batch['f1'].append(f1.item())

            elif metric == 'auc':
                # Placeholder AUC (use sklearn.metrics.roc_auc_score for real AUC)
                if predictions.dim() > 1 and predictions.size(1) == 2:
                    probs = torch.softmax(predictions, dim=1)[:, 1]
                else:
                    probs = torch.sigmoid(predictions)
                self.current_batch['auc'].append(0.5)  # placeholder value

            elif metric == 'mae':
                mae = torch.abs(predictions - targets).mean().item()
                self.current_batch['mae'].append(mae)

            elif metric == 'rmse':
                mse = ((predictions - targets) ** 2).mean().item()
                rmse = float(np.sqrt(mse))
                self.current_batch['rmse'].append(rmse)

    def compute(self) -> Dict[str, float]:
        """
        Compute the mean value for each tracked metric in the current batch.

        Returns:
            A dict mapping metric names to values
        """
        results: Dict[str, float] = {}
        for metric in self.metrics:
            if metric in self.current_batch and self.current_batch[metric]:
                results[metric] = float(np.mean(self.current_batch[metric]))
        return results

    def reset(self) -> None:
        """Clear current batch statistics."""
        self.current_batch.clear()

    def epoch_end(self) -> None:
        """Push the current batch stats to history and reset batch stats."""
        epoch_metrics = self.compute()
        for metric, value in epoch_metrics.items():
            self.history[metric].append(value)
        self.reset()

    def plot_history(self, save_path: Optional[str] = None) -> None:
        """Plot curves for all tracked metrics in history.

        If save_path is provided, the figure will be saved to that path.
        """
        if not self.history:
            logger.warning("No history to plot")
            return

        fig, axes = plt.subplots(len(self.history), 1, figsize=(10, 4 * len(self.history)))
        if len(self.history) == 1:
            axes = [axes]

        for ax, (metric, values) in zip(axes, self.history.items()):
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, 'b-', label=metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} History')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metric plots saved to {save_path}")
        plt.close()


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: Model instance
        trainable_only: If True, count only parameters with requires_grad=True

    Returns:
        The number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def save_config(config: Any, path: str) -> None:
    """
    Save a configuration object or dict to a JSON file.

    Args:
        config: The configuration (object with __dict__ or a dict)
        path: Output JSON path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict if needed
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config

    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    logger.info(f"Configuration saved to {path}")


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    optimizer_type: str = 'adamw',
    weight_decay: float = 0.01,
    **kwargs,
) -> Optimizer:
    """
    Create an optimizer for the given model parameters.

    Args:
        model: Model instance
        lr: Learning rate
        optimizer_type: One of {'adam', 'adamw', 'sgd', 'rmsprop'}
        weight_decay: Weight decay factor
        **kwargs: Extra keyword arguments forwarded to the optimizer

    Returns:
        The created optimizer instance
    """
    params = model.parameters()

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            **kwargs,
        )
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    logger.info(f"Created {optimizer_type} optimizer, lr={lr}")
    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    scheduler_type: str = 'cosine',
    num_warmup_steps: int = 0,
    **kwargs,
) -> _LRScheduler:
    """
    Create a learning-rate scheduler.

    Args:
        optimizer: Optimizer instance
        num_training_steps: Total number of training steps
        scheduler_type: One of {'cosine', 'linear', 'constant', 'exponential'}
        num_warmup_steps: Warmup steps for 'linear' schedule
        **kwargs: Extra kwargs forwarded to scheduler constructors

    Returns:
        The created scheduler instance
    """
    if scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, **kwargs)

    elif scheduler_type == 'linear':
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step):
            if step < num_warmup_steps:
                return float(step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        scheduler = LambdaLR(optimizer, lr_lambda, **kwargs)

    elif scheduler_type == 'constant':
        from torch.optim.lr_scheduler import LambdaLR

        scheduler = LambdaLR(optimizer, lambda _: 1.0, **kwargs)

    elif scheduler_type == 'exponential':
        from torch.optim.lr_scheduler import ExponentialLR

        scheduler = ExponentialLR(optimizer, gamma=kwargs.get('gamma', 0.95))

    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    logger.info(f"Created {scheduler_type} learning-rate scheduler")
    return scheduler

