"""
Train KumoRFM on RelBench datasets
"""

import argparse
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import time
from typing import Dict, Any, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

# Ensure project root is importable when running this file directly
import sys
from pathlib import Path as _PathForSys
_HERE = _PathForSys(__file__).resolve()
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.model_config import KumoRFMConfig, ExperimentConfig
from models.kumorfm import KumoRFM
from training.trainer import KumoRFMTrainer
from utils.training_utils import set_random_seed
from relbdata.adapter import (
    RelBenchAdapter,
    get_database_schema_from_relbench
)
from sampling.context_label_table import InContextLabelTable, ForwardLabelSampler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RelBenchDataset(Dataset):
    """
    PyTorch Dataset wrapper for RelBench datasets
    """

    def __init__(self,
                 graph: 'TemporalHeterogeneousGraph',
                 entities: List[Tuple[str, int]],
                 labels: np.ndarray,
                 timestamps: List[datetime],
                 task_config: 'TaskConfig'):
        self.graph = graph
        self.entities = entities
        # Normalize labels based on task type
        self.task_config = task_config
        labels_array = np.asarray(labels)
        if task_config.task_type == 'classification':
            # Always remap labels to contiguous indices [0..K-1]
            labels_str = labels_array.astype(str)
            uniques, inverse = np.unique(labels_str, return_inverse=True)
            self.class_names = uniques.tolist()
            self.labels = torch.tensor(inverse, dtype=torch.long)
        else:
            # Regression: ensure float tensor; coerce non-numeric to NaN then fill 0
            try:
                y = pd.to_numeric(labels_array, errors='coerce').astype(float)
            except Exception:
                y = np.array([float(x) if str(x).replace('.', '', 1).isdigit() else np.nan for x in labels_array], dtype=float)
            y = np.nan_to_num(y, nan=0.0)
            self.labels = torch.tensor(y, dtype=torch.float)
        self.timestamps = timestamps

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, idx):
        return {
            'entity': self.entities[idx],
            'label': self.labels[idx],
            'timestamp': self.timestamps[idx]
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Batch processing function"""
    return {
        'entities': [item['entity'] for item in batch],
        'labels': torch.stack([item['label'] for item in batch]),
        'timestamps': [item['timestamp'] for item in batch]
    }


class RelBenchTrainer:
    """
    RelBench specialized trainer
    """

    def __init__(self,
                 model: KumoRFM,
                 graph: 'TemporalHeterogeneousGraph',
                 config: KumoRFMConfig,
                 experiment_config: ExperimentConfig,
                 task_config: 'TaskConfig',
                 device: torch.device):
        self.model = model
        self.graph = graph
        self.config = config
        self.experiment_config = experiment_config
        self.task_config = task_config
        self.device = device

        # Create base trainer
        self.base_trainer = KumoRFMTrainer(
            model, config, experiment_config, device
        )
        self.best_checkpoint_path = None

        # Set task configuration
        self.model.set_task_config(task_config)

        # Loss function
        self.criterion = self._get_criterion()

    def _get_criterion(self):
        """Get loss function"""
        if self.task_config.task_type == 'classification':
            # Use CrossEntropy for all classification (binary and multi-class)
            return torch.nn.CrossEntropyLoss()
        else:
            return torch.nn.MSELoss()

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train model"""
        logger.info("Starting training...")

        # Timing trackers
        self.epoch_times: List[float] = []
        self.train_start_iso = datetime.now().isoformat()
        t0_total = time.time()

        best_val_metric = float('inf') if self.task_config.task_type == 'regression' else 0.0

        for epoch in range(self.experiment_config.num_epochs):
            t0_epoch = time.time()
            # Training phase
            train_loss = self._train_epoch(train_loader, epoch)

            # Validation phase
            val_loss, val_metric = self._validate(val_loader)

            # Logging
            logger.info(f"Epoch {epoch + 1}/{self.experiment_config.num_epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Val Metric: {val_metric:.4f}")

            # Epoch timing
            epoch_time = time.time() - t0_epoch
            self.epoch_times.append(epoch_time)
            logger.info(f"  Epoch Time: {epoch_time:.2f}s")

            # Save best model
            if self._is_better(val_metric, best_val_metric):
                best_val_metric = val_metric
                self._save_checkpoint(epoch, val_metric)
                logger.info(f"  New best model!")

            # Early stopping check
            if self.base_trainer.early_stopping(val_loss):
                logger.info("Early stopping triggered, stopping training")
                break

        # Total timing
        self.total_time_sec = time.time() - t0_total
        self.train_end_iso = datetime.now().isoformat()
        logger.info(f"Training completed in {self.total_time_sec:.2f}s. Best validation metric: {best_val_metric:.4f}")

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Prepare data
            batch_data = self._prepare_batch(batch)

            # Zero gradients
            self.base_trainer.optimizer.zero_grad()

            # Forward pass
            loss = 0.0
            for i in range(len(batch_data['entities'])):
                output = self.model(
                    self.graph,
                    batch_data['entities'][i],
                    batch_data['timestamps'][i],
                    self.task_config,
                    context_strategy='mixed',
                    num_context=5
                )

                # Extract prediction
                pred = self._extract_prediction(output)

                # Calculate loss
                if pred is not None:
                    target = batch_data['labels'][i:i+1]
                    if self.task_config.task_type == 'classification':
                        target = target.long()

                    item_loss = self.criterion(pred, target)
                    loss += item_loss

            # Average loss
            loss = loss / len(batch_data['entities'])

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )

            # Update parameters
            self.base_trainer.optimizer.step()
            self.base_trainer.scheduler.step()

            # Logging
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(train_loader)

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                batch_data = self._prepare_batch(batch)

                loss = 0.0
                for i in range(len(batch_data['entities'])):
                    output = self.model(
                        self.graph,
                        batch_data['entities'][i],
                        batch_data['timestamps'][i],
                        self.task_config,
                        context_strategy='mixed',
                        num_context=5
                    )

                    # Extract prediction
                    pred = self._extract_prediction(output)

                    if pred is not None:
                        target = batch_data['labels'][i:i+1]
                        if self.task_config.task_type == 'classification':
                            target = target.long()

                        item_loss = self.criterion(pred, target)
                        loss += item_loss

                        # Collect predictions
                        if self.task_config.task_type == 'classification':
                            # If logits vector, use argmax; if scalar logit, use sigmoid threshold
                            if pred.dim() > 1 and pred.shape[-1] > 1:
                                pred_class = pred.argmax(dim=-1).float()
                            else:
                                pred_class = (torch.sigmoid(pred) > 0.5).float()
                            all_preds.append(pred_class.item())
                        else:
                            all_preds.append(pred.item())

                        all_labels.append(batch_data['labels'][i].item())

                loss = loss / len(batch_data['entities'])
                total_loss += loss.item()

        # Calculate metric
        avg_loss = total_loss / len(val_loader)
        metric = self._calculate_metric(all_preds, all_labels)

        return avg_loss, metric

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model"""
        logger.info("Evaluating model...")

        test_loss, test_metric = self._validate(test_loader)

        results = {
            'test_loss': test_loss,
            'test_metric': test_metric
        }

        # Add specific metric based on task type
        if self.task_config.task_type == 'classification':
            results['accuracy'] = test_metric
        else:
            results['mae'] = test_metric

        return results

    def _prepare_batch(self, batch: Dict) -> Dict:
        """Prepare batch data"""
        return {
            'entities': batch['entities'],
            'labels': batch['labels'].to(self.device),
            'timestamps': batch['timestamps']
        }

    def _extract_prediction(self, output: Dict) -> Optional[torch.Tensor]:
        """Extract prediction tensor from output"""
        if 'predictions' in output:
            pred = output['predictions']
        elif 'probabilities' in output:
            pred = torch.tensor(output['probabilities'])
        elif 'predicted_value' in output:
            pred = torch.tensor([[output['predicted_value']]], device=self.device)
        else:
            return None

        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred, device=self.device)

        if pred.dim() == 0:
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 1:
            pred = pred.unsqueeze(0)

        return pred

    def _calculate_metric(self, preds: List[float], labels: List[float]) -> float:
        """Calculate evaluation metric"""
        preds = np.array(preds)
        labels = np.array(labels)

        if self.task_config.task_type == 'classification':
            # Accuracy
            return (preds == labels).mean()
        else:
            # MAE
            return np.abs(preds - labels).mean()

    def _is_better(self, current: float, best: float) -> bool:
        """Determine if the current metric is better than the best"""
        if self.task_config.task_type == 'classification':
            return current > best
        else:
            return current < best

    def _save_checkpoint(self, epoch: int, metric: float):
        """Save checkpoint"""
        checkpoint_path = Path(self.experiment_config.save_dir) / f'best_model.pt'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.base_trainer.optimizer.state_dict(),
            'metric': metric,
            'config': self.config,
            'task_config': self.task_config
        }

        torch.save(checkpoint, checkpoint_path)
        self.best_checkpoint_path = checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='Train KumoRFM on RelBench')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='amazon',
                       choices=['amazon', 'stack', 'f1', 'trial', 'avito', 'event', 'hm'],
                       help='RelBench dataset name')
    parser.add_argument('--task', type=str, default='auto', required=False,
                       help="Task name (e.g., user-churn). Use 'auto' to infer/fallback.")

    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of layers')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./relbench_outputs',
                       help='Output directory')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run with a tiny synthetic dataset (no RelBench needed)')

    args = parser.parse_args()

    # Set device and random seed
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    set_random_seed(args.seed)
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir) / f"{args.dataset}_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Prepare in-context label table + forward sampler
    label_table = InContextLabelTable()
    forward_sampler = ForwardLabelSampler(label_table)

    # Optionally run a tiny synthetic pipeline to validate end-to-end
    if args.dry_run:
        logger.info("Dry-run enabled: using a tiny synthetic dataset")

        # Build a tiny graph with one node type and no edges
        from data.temporal_graph import TemporalHeterogeneousGraph
        graph = TemporalHeterogeneousGraph()
        num_nodes = 20
        features = torch.randn(num_nodes, args.hidden_dim)
        graph.add_node_type('user', num_nodes, features=features)

        # Minimal database schema
        database_schema = {
            'user': {
                'feature': 'numerical',
            }
        }

        # Task: binary classification on a single entity type
        from config.model_config import TaskConfig
        task_config = TaskConfig(task_type='classification', num_classes=2, target_column='label')

        # Create synthetic splits
        all_entities = [('user', i) for i in range(num_nodes)]
        all_labels = (torch.rand(num_nodes) > 0.5).float().numpy()
        base_time = datetime.now()
        all_times = [base_time for _ in range(num_nodes)]

        idx_train = list(range(0, int(num_nodes*0.6)))
        idx_val = list(range(int(num_nodes*0.6), int(num_nodes*0.8)))
        idx_test = list(range(int(num_nodes*0.8), num_nodes))

        def subset(indices):
            return {
                'entities': [all_entities[i] for i in indices],
                'labels': [all_labels[i] for i in indices],
                'timestamps': [all_times[i] for i in indices],
            }

        data_splits = {
            'train': subset(idx_train),
            'val': subset(idx_val),
            'test': subset(idx_test),
        }
    else:
        # Load RelBench data (robust to dataset variants and availability)
        logger.info(f"Loading RelBench dataset: {args.dataset}")
        adapter = RelBenchAdapter(args.dataset)

        def try_prepare(_adapter):
            _dataset = _adapter.load_dataset()
            _db = _adapter.convert_database()
            _graph = _adapter.build_temporal_graph()
            return _dataset, _db, _graph

        candidates = [args.dataset, f"rel-{args.dataset}", "trial", "rel-trial", "stack", "rel-stack", "f1", "rel-f1"]
        dataset = None
        for cand in candidates:
            try:
                name = cand.replace("rel-", "")
                adapter = RelBenchAdapter(name)
                dataset, database, graph = try_prepare(adapter)
                logger.info(f"Prepared dataset '{cand}' successfully")
                break
            except Exception as e:
                logger.warning(f"Dataset candidate '{cand}' failed: {e}")
                continue
        if dataset is None:
            raise RuntimeError("Failed to prepare any RelBench dataset. Ensure data is available locally.")

        # Get database schema
        database_schema = get_database_schema_from_relbench(dataset)

        # Get task configuration
        task_config = adapter.get_task_config(args.task)

        # Get data splits
        data_splits = adapter.get_train_test_split(args.task)

    # Populate in-context label table (train + val splits as historical contexts)
    forward_sampler.ingest_relbench_split(data_splits['train'], metadata={'split': 'train'})
    if 'val' in data_splits:
        forward_sampler.ingest_relbench_split(data_splits['val'], metadata={'split': 'val'})
    label_table.finalize()

    # Create datasets
    train_dataset = RelBenchDataset(
        graph,
        data_splits['train']['entities'],
        data_splits['train']['labels'],
        data_splits['train']['timestamps'],
        task_config
    )

    val_dataset = RelBenchDataset(
        graph,
        data_splits['val']['entities'],
        data_splits['val']['labels'],
        data_splits['val']['timestamps'],
        task_config
    )

    test_dataset = RelBenchDataset(
        graph,
        data_splits['test']['entities'],
        data_splits['test']['labels'],
        data_splits['test']['timestamps'],
        task_config
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # If classification, infer number of classes from data to avoid head-size mismatches
    if task_config.task_type == 'classification':
        try:
            # Collect labels from train/val/test
            all_labels = torch.cat([
                train_dataset.labels.view(-1),
                val_dataset.labels.view(-1),
                test_dataset.labels.view(-1)
            ])
            inferred_num_classes = int(all_labels.max().item()) + 1 if all_labels.numel() > 0 else 2
            if not task_config.num_classes or task_config.num_classes < inferred_num_classes:
                logger.info(f"Adjusting num_classes from {task_config.num_classes} to {inferred_num_classes}")
                task_config.num_classes = inferred_num_classes
        except Exception as e:
            logger.warning(f"Could not infer number of classes from data: {e}")

    # Create model configuration
    model_config = KumoRFMConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    experiment_config = ExperimentConfig(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        save_dir=str(output_dir / 'checkpoints'),
        log_dir=str(output_dir / 'logs')
    )

    # Create model
    logger.info("Creating KumoRFM model...")
    model = KumoRFM(model_config, database_schema).to(device)
    if hasattr(model, 'context_sampler'):
        model.context_sampler.attach_label_table(label_table)

    # Create trainer
    trainer = RelBenchTrainer(
        model, graph, model_config, experiment_config, task_config, device
    )

    # Train model
    trainer.train(train_loader, val_loader)

    # Evaluate model
    test_results = trainer.evaluate(test_loader)

    # Save results
    results = {
        'dataset': args.dataset,
        'task': args.task,
        'config': vars(args),
        'test_results': test_results,
        'model_config': model_config.__dict__,
        'task_config': task_config.__dict__,
        'best_model_path': str(trainer.best_checkpoint_path) if trainer.best_checkpoint_path else None,
        # Timing info
        'epoch_times_sec': getattr(trainer, 'epoch_times', []),
        'total_training_time_sec': getattr(trainer, 'total_time_sec', None),
        'training_started_at': getattr(trainer, 'train_start_iso', None),
        'training_finished_at': getattr(trainer, 'train_end_iso', None),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nTest Results:")
    for metric, value in test_results.items():
        logger.info(f"  {metric}: {value:.4f}")

    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
