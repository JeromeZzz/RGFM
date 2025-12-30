"""
RelBench Integration Module for KumoRFM

This module provides integration between KumoRFM and RelBench datasets,
enabling training and evaluation on relational benchmarks.

Main Components:
- RelBenchAdapter: Converts RelBench datasets to KumoRFM format
- RelBenchTrainer: Specialized trainer for RelBench tasks  
- RelBenchDataset: PyTorch dataset wrapper for RelBench data
- Analysis tools: Comprehensive evaluation and visualization utilities

Usage:
    from kumorfm.relbench import RelBenchAdapter, train_on_relbench
    
    # Load and convert dataset
    adapter = RelBenchAdapter("amazon")
    dataset = adapter.load_dataset()
    graph = adapter.build_temporal_graph()
    
    # Train model
    results = train_on_relbench(
        dataset_name="amazon",
        task_name="user-churn", 
        config=config
    )
"""

from .adapter import (
    RelBenchAdapter,
    RelBenchDatabase, 
    RelBenchGraphConverter,
    get_database_schema_from_relbench
)

# Training and analysis components are optional on import to avoid
# pulling heavy or optional dependencies when not needed.
try:
    from .train_on_relbench import (
        RelBenchDataset,
        RelBenchTrainer,
        collate_fn,
        main as train_on_relbench
    )
except Exception:  # keep import-time robust for adapter usage
    RelBenchDataset = None
    RelBenchTrainer = None
    collate_fn = None
    train_on_relbench = None

try:
    from .analyze_results import (
        load_results,
        load_all_results,
        create_comparison_table,
        plot_metric_comparison,
        plot_hyperparameter_impact,
        analyze_training_efficiency,
        generate_report
    )
except Exception:
    load_results = None
    load_all_results = None
    create_comparison_table = None
    plot_metric_comparison = None
    plot_hyperparameter_impact = None
    analyze_training_efficiency = None
    generate_report = None

# Version and metadata
__version__ = "0.1.0"
__author__ = "Zhao"

# Public API
__all__ = [
    # Core adapter classes
    "RelBenchAdapter",
    "RelBenchDatabase", 
    "RelBenchGraphConverter",
    "get_database_schema_from_relbench",
    
    # Training components
    # Training components (may be None if optional import failed)
    "RelBenchDataset",
    "RelBenchTrainer",
    "collate_fn",
    "train_on_relbench",
    
    # Analysis utilities
    # Analysis utilities (may be None if optional import failed)
    "load_results",
    "load_all_results",
    "create_comparison_table",
    "plot_metric_comparison",
    "plot_hyperparameter_impact",
    "analyze_training_efficiency",
    "generate_report",
]

# Supported RelBench datasets
SUPPORTED_DATASETS = [
    "amazon",
    "stack", 
    "f1",
    "trial",
    "avito",
    "event",
    "hm"
]

# Common task types per dataset
DATASET_TASKS = {
    "amazon": ["user-churn", "item-sales", "user-ltv"],
    "stack": ["user-engagement", "question-quality"],
    "f1": ["driver-performance", "team-performance"], 
    "trial": ["patient-outcome", "treatment-response"],
    "avito": ["user-activity", "item-popularity"],
    "event": ["event-attendance", "user-retention"],
    "hm": ["customer-purchase", "product-recommendation"]
}

def get_available_datasets():
    """Get list of supported RelBench datasets."""
    return SUPPORTED_DATASETS.copy()

def get_dataset_tasks(dataset_name: str):
    """Get available tasks for a specific dataset."""
    return DATASET_TASKS.get(dataset_name, [])

def validate_dataset_task(dataset_name: str, task_name: str):
    """Validate if dataset and task combination is supported."""
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported: {SUPPORTED_DATASETS}")
    
    expected_tasks = DATASET_TASKS.get(dataset_name, [])
    if expected_tasks and task_name not in expected_tasks:
        raise ValueError(f"Task '{task_name}' may not be available for dataset '{dataset_name}'. "
                        f"Common tasks: {expected_tasks}")
    
    return True
