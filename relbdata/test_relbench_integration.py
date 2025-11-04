"""
RelBench Integration Test
Test KumoRFM integration with RelBench datasets
"""

import sys
import torch
from datetime import datetime
import logging
print("Python path:", sys.path)
print("Current working directory:", __file__)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_relbench_import():
    """Test RelBench import"""
    logger.info("Testing RelBench import...")
    try:
        import relbench
        logger.info(f"RelBench version: {relbench.__version__ if hasattr(relbench, '__version__') else 'unknown'}")
        return True
    except ImportError as e:
        logger.error(f"Failed to import RelBench: {e}")
        logger.error("  Install with: pip install relbench")
        return False


def test_dataset_loading():
    """Test dataset loading for current RelBench version"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("\nTesting dataset loading...")

    try:
        from relbench.datasets import get_dataset

        # Load dataset without downloading
        dataset = get_dataset("rel-trial", download=False)
        logger.info(f"? Successfully loaded dataset: {dataset.__class__.__name__}")

        # Current version does not have .tasks
        logger.info(f"Entity table: {dataset.entity_table}")
        logger.info(f"Target column: {dataset.target_col}")
        logger.info(f"Validation timestamp: {dataset.val_timestamp}")
        logger.info(f"Test timestamp: {dataset.test_timestamp}")

        # Show available tables
        db = dataset.get_db()
        logger.info(f"Tables in dataset: {list(db.table_dict.keys())}")


        # Optional: try downloading (if method exists)
        if hasattr(dataset, "download"):
            try:
                dataset.download()
                logger.info("? Successfully downloaded dataset")
            except Exception as e:
                logger.warning(f"Dataset download failed (expected in test): {e}")

        return True

    except Exception as e:
        logger.error(f"? Dataset loading failed: {e}")
        return False




def test_adapter():
    """Test KumoRFM Adapter"""
    logger.info("\nTesting KumoRFM adapter...")
    try:
        from relbdata.adapter import RelBenchAdapter

        # Create adapter
        adapter = RelBenchAdapter("amazon")
        logger.info("  Adapter created")

        # Load dataset
        dataset = adapter.load_dataset()
        logger.info(f"  Dataset loaded: {dataset.name}")

        # Convert database
        database = adapter.convert_database()
        logger.info(f"  Database converted: {len(database.tables)} tables")

        # Build graph
        logger.info("  Building temporal graph...")
        graph = adapter.build_temporal_graph()
        logger.info(f"  Graph built: {graph.total_nodes} nodes")

        return True
    except Exception as e:
        logger.error(f"  Adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation"""
    logger.info("\nTesting model creation...")
    try:
        from config.model_config import KumoRFMConfig
        from models.kumorfm import KumoRFM
        from relbdata.adapter import get_database_schema_from_relbench
        from relbdata.datasets import get_dataset

        # Get database schema
        dataset = get_dataset("amazon", download=False)
        database_schema = get_database_schema_from_relbench(dataset)
        logger.info(f"  Schema loaded: {len(database_schema)} tables")

        # Create model config
        config = KumoRFMConfig(
            hidden_dim=64,  # Reduced dimension for testing
            num_layers=1,
            num_heads=2
        )

        # Create model
        model = KumoRFM(config, database_schema)
        logger.info(f"  Model created")

        # Check parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Total parameters: {total_params:,}")

        return True
    except Exception as e:
        logger.error(f"  Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_forward():
    """Test simple forward pass"""
    logger.info("\nTesting simple forward pass...")
    try:
        from relbdata.adapter import RelBenchAdapter, get_database_schema_from_relbench
        from config.model_config import KumoRFMConfig, TaskConfig
        from models.kumorfm import KumoRFM

        # Load data
        adapter = RelBenchAdapter("amazon")
        dataset = adapter.load_dataset()
        database = adapter.convert_database()
        graph = adapter.build_temporal_graph()

        # Get task config
        task_name = "user-churn"
        task_config = adapter.get_task_config(task_name)
        logger.info(f"  Task type: {task_config.task_type}")

        # Create model
        database_schema = get_database_schema_from_relbench(dataset)
        config = KumoRFMConfig(hidden_dim=64, num_layers=1, num_heads=2)
        model = KumoRFM(config, database_schema)
        model.eval()

        # Simple forward pass
        with torch.no_grad():
            output = model(
                graph,
                target_entity=('user', 0),  # Example entity 'user'
                prediction_time=datetime.now(),
                task_config=task_config,
                num_context=2
            )

        logger.info(f"  Forward pass successful")
        logger.info(f"  Output keys: {list(output.keys())}")

        return True
    except Exception as e:
        logger.error(f"  Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("KumoRFM-RelBench Integration Test")
    logger.info("=" * 60)

    tests = [
        ("RelBench Import", test_relbench_import),
        ("Dataset Loading", test_dataset_loading),
        ("Adapter Test", test_adapter),
        ("Model Creation", test_model_creation),
        ("Simple Forward", test_simple_forward)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} encountered an unexpected error: {e}")
            results[test_name] = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    logger.info("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_status in results.items():
        status = "PASSED" if passed_status else "FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\nAll tests passed successfully.")
        logger.info("\nSuggestion:")
        logger.info("python relbench/train_on_relbench.py --dataset amazon --task user-churn")
    else:
        logger.error("\nSome tests failed. Check the logs for details.")

    return passed == total


if __name__ == '__main__':
    sys.exit(0 if main() else 1)