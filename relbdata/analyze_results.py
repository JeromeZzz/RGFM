"""
Analyze RelBench training results
Visualize and compare results from different experiments
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Any
import numpy as np

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_results(result_dir: Path) -> Dict[str, Any]:
    """Load single experiment results"""
    results_file = result_dir / 'results.json'
    config_file = result_dir / 'config.json'

    if not results_file.exists():
        return None

    with open(results_file, 'r') as f:
        results = json.load(f)

    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        results['config'] = config

    results['exp_name'] = result_dir.name
    return results


def load_all_results(base_dir: str) -> List[Dict[str, Any]]:
    """Load all experiment results"""
    base_path = Path(base_dir)
    all_results = []

    for exp_dir in base_path.iterdir():
        if exp_dir.is_dir():
            result = load_results(exp_dir)
            if result:
                all_results.append(result)

    return all_results


def create_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create comparison table"""
    data = []

    for r in results:
        row = {
            'Experiment': r['exp_name'],
            'Dataset': r.get('dataset', 'unknown'),
            'Task': r.get('task', 'unknown'),
            'Hidden Dim': r.get('config', {}).get('hidden_dim', 'N/A'),
            'Num Layers': r.get('config', {}).get('num_layers', 'N/A'),
            'Batch Size': r.get('config', {}).get('batch_size', 'N/A'),
            'Learning Rate': r.get('config', {}).get('lr', 'N/A'),
        }

        # Add test results
        test_results = r.get('test_results', {})
        for metric, value in test_results.items():
            row[f'Test {metric}'] = f"{value:.4f}" if isinstance(value, (int, float)) else value

        data.append(row)

    return pd.DataFrame(data)


def plot_metric_comparison(results: List[Dict[str, Any]], metric: str = 'accuracy'):
    """Plot metric comparison chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data
    datasets = []
    tasks = []
    values = []
    experiments = []

    for r in results:
        test_results = r.get('test_results', {})
        if metric in test_results:
            datasets.append(r.get('dataset', 'unknown'))
            tasks.append(r.get('task', 'unknown'))
            values.append(test_results[metric])
            experiments.append(r['exp_name'])

    if not values:
        print(f"No data found for metric {metric}")
        return

    # Bar chart grouped by dataset
    df = pd.DataFrame({
        'Dataset': datasets,
        'Task': tasks,
        'Value': values,
        'Experiment': experiments
    })

    # Left plot: grouped by dataset
    dataset_groups = df.groupby('Dataset')['Value'].agg(['mean', 'std', 'count'])
    dataset_groups.plot(kind='bar', y='mean', yerr='std', ax=ax1, legend=False)
    ax1.set_title(f'{metric.capitalize()} by Dataset')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel(metric.capitalize())
    ax1.tick_params(axis='x', rotation=45)

    # Right plot: grouped by task
    task_groups = df.groupby('Task')['Value'].agg(['mean', 'std', 'count'])
    task_groups.plot(kind='bar', y='mean', yerr='std', ax=ax2, legend=False)
    ax2.set_title(f'{metric.capitalize()} by Task')
    ax2.set_xlabel('Task')
    ax2.set_ylabel(metric.capitalize())
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_hyperparameter_impact(results: List[Dict[str, Any]]):
    """Plot hyperparameter impact analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Prepare data
    data = []
    for r in results:
        config = r.get('config', {})
        test_results = r.get('test_results', {})

        # Get main metric (accuracy or MAE)
        metric_value = test_results.get('accuracy', test_results.get('mae', None))
        if metric_value is None:
            continue

        data.append({
            'hidden_dim': config.get('hidden_dim', 256),
            'num_layers': config.get('num_layers', 4),
            'lr': config.get('lr', 0.0001),
            'batch_size': config.get('batch_size', 32),
            'metric': metric_value
        })

    if not data:
        print("Not enough data for hyperparameter analysis")
        return

    df = pd.DataFrame(data)

    # Plot impact of each hyperparameter
    hyperparams = ['hidden_dim', 'num_layers', 'lr', 'batch_size']

    for i, param in enumerate(hyperparams):
        if i < len(axes):
            # Create box plot
            unique_values = sorted(df[param].unique())

            if len(unique_values) > 1:
                data_by_param = [df[df[param] == val]['metric'].values
                                 for val in unique_values]

                axes[i].boxplot(data_by_param, labels=unique_values)
                axes[i].set_xlabel(param.replace('_', ' ').title())
                axes[i].set_ylabel('Performance Metric')
                axes[i].set_title(f'Impact of {param.replace("_", " ").title()}')

                # Add trend line
                if len(unique_values) > 2:
                    means = [np.mean(d) if len(d) > 0 else 0 for d in data_by_param]
                    axes[i].plot(range(1, len(unique_values) + 1), means, 'r--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('hyperparameter_impact.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_training_efficiency(results: List[Dict[str, Any]]):
    """Analyze training efficiency"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data
    data = []
    for r in results:
        config = r.get('config', {})
        test_results = r.get('test_results', {})

        # Calculate model complexity (simplified)
        hidden_dim = config.get('hidden_dim', 256)
        num_layers = config.get('num_layers', 4)
        complexity = hidden_dim * num_layers

        # Get performance
        performance = test_results.get('accuracy', test_results.get('mae', 0))

        data.append({
            'dataset': r.get('dataset', 'unknown'),
            'complexity': complexity,
            'performance': performance,
            'batch_size': config.get('batch_size', 32)
        })

    df = pd.DataFrame(data)

    # Left plot: Complexity vs Performance
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        ax1.scatter(subset['complexity'], subset['performance'],
                    label=dataset, s=100, alpha=0.6)

    ax1.set_xlabel('Model Complexity (hidden_dim × num_layers)')
    ax1.set_ylabel('Performance')
    ax1.set_title('Model Complexity vs Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Batch Size vs Performance
    batch_sizes = sorted(df['batch_size'].unique())
    performances = [df[df['batch_size'] == bs]['performance'].mean()
                    for bs in batch_sizes]

    ax2.plot(batch_sizes, performances, 'o-', markersize=10, linewidth=2)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Average Performance')
    ax2.set_title('Batch Size vs Performance')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_report(results: List[Dict[str, Any]], output_file: str = 'report.html'):
    """Generate HTML report"""
    html_content = f"""
    <html>
    <head>
        <title>KumoRFM RelBench Experiment Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; color: #2e7d32; }}
            h2 {{ color: #1976d2; }}
        </style>
    </head>
    <body>
        <h1>KumoRFM RelBench Experiment Report</h1>
        <p>Generation Time: {pd.Timestamp.now()}</p>
        <p>Number of Experiments: {len(results)}</p>

        <h2>Experiment Summary</h2>
    """

    # Add summary statistics
    datasets = [r.get('dataset', 'unknown') for r in results]
    tasks = [r.get('task', 'unknown') for r in results]

    html_content += f"""
        <ul>
            <li>Datasets: {', '.join(set(datasets))}</li>
            <li>Tasks: {', '.join(set(tasks))}</li>
        </ul>

        <h2>Detailed Results</h2>
    """

    # Add comparison table
    df = create_comparison_table(results)
    html_content += df.to_html(index=False, classes='results-table')

    # Find best results
    best_results = {}
    for r in results:
        key = f"{r.get('dataset', 'unknown')}_{r.get('task', 'unknown')}"
        test_results = r.get('test_results', {})

        # Get main metric
        if 'accuracy' in test_results:
            metric = 'accuracy'
            value = test_results['accuracy']
            is_better = lambda x, y: x > y
        elif 'mae' in test_results:
            metric = 'mae'
            value = test_results['mae']
            is_better = lambda x, y: x < y
        else:
            continue

        if key not in best_results or is_better(value, best_results[key]['value']):
            best_results[key] = {
                'experiment': r['exp_name'],
                'metric': metric,
                'value': value,
                'config': r.get('config', {})
            }

    html_content += """
        <h2>Best Results</h2>
        <table>
            <tr>
                <th>Dataset-Task</th>
                <th>Best Experiment</th>
                <th>Metric</th>
                <th>Value</th>
                <th>Configuration</th>
            </tr>
    """

    for key, best in best_results.items():
        config_str = f"Hidden: {best['config'].get('hidden_dim', 'N/A')}, " \
                     f"Layers: {best['config'].get('num_layers', 'N/A')}, " \
                     f"LR: {best['config'].get('lr', 'N/A')}"

        html_content += f"""
            <tr>
                <td>{key}</td>
                <td>{best['experiment']}</td>
                <td>{best['metric']}</td>
                <td class="metric">{best['value']:.4f}</td>
                <td>{config_str}</td>
            </tr>
        """

    html_content += """
        </table>

        <h2>Suggestions</h2>
        <ul>
            <li>If accuracy is lower than expected, try increasing epochs or adjusting learning rate</li>
            <li>For large datasets, increasing batch size can improve training speed</li>
            <li>Consider using a deeper model (increase num_layers) to capture complex patterns</li>
        </ul>
    </body>
    </html>
    """

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze RelBench training results')
    parser.add_argument('--results-dir', type=str, default='./relbench_outputs',
                        help='Results directory')
    parser.add_argument('--metric', type=str, default='accuracy',
                        help='Metric to analyze')
    parser.add_argument('--output-dir', type=str, default='./analysis',
                        help='Output directory')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load all results
    print(f"Loading results from {args.results-dir}...")
    results = load_all_results(args.results_dir)

    if not results:
        print("No results files found")
        return

    print(f"Found {len(results)} experiment results")

    # Create comparison table
    print("\nCreating comparison table...")
    df = create_comparison_table(results)
    print(df)
    df.to_csv(output_dir / 'comparison_table.csv', index=False)

    # Plot charts
    print("\nGenerating visualizations...")
    plot_metric_comparison(results, args.metric)
    plot_hyperparameter_impact(results)
    analyze_training_efficiency(results)

    # Generate report
    print("\nGenerating HTML report...")
    generate_report(results, output_dir / 'report.html')

    print(f"\nAll analysis results saved to: {output_dir}")


if __name__ == '__main__':
    main()