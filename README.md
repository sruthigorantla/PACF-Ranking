# Active Fair Ranking

This repository contains the implementation of active fair ranking algorithms, focusing on fair ranking with active learning approaches. The project implements various algorithms including Beat the Pivot and baseline methods for fair ranking.

## Project Description

This project implements algorithms for fair ranking with active learning, where the goal is to learn a fair ranking of items while minimizing the number of pairwise comparisons needed. The implementation includes:

- Beat the Pivot algorithm
- Baseline ranking algorithms
- Support for multiple datasets (COMPAS, German Credit, etc.)
- Fairness-aware ranking with protected attributes
- Evaluation metrics including Kendall Tau ranking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/active-fair-ranking.git
cd active-fair-ranking
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Dependencies

The project requires the following Python packages:
- matplotlib
- omegaconf
- pandas
- scipy
- seaborn
- tqdm

These dependencies will be automatically installed when you install the package.

## Project Structure

```
active-fair-ranking/
├── src/
│   └── active_fair_ranking/
│       └── algorithms/
│           ├── beat_the_pivot.py
│           ├── baseline.py
│           ├── data.py
│           ├── find_the_pivot.py
│           ├── merge_rankings.py
│           ├── pairwise.py
│           ├── ranking_algorithms.py
│           └── utils.py
├── scripts/
│   ├── run_beat_the_pivot.py
│   ├── run_baseline.py
│   └── plot_results.py
├── data/
│   └── COMPAS/
└── setup.py
```

## Usage

### Running Experiments

1. Navigate to the scripts directory:
```bash
cd scripts
```

2. Run the Beat the Pivot algorithm:
```bash
python run_beat_the_pivot.py
```

3. Run baseline experiments:
```bash
python run_baseline.py --exp-id <experiment_id>
```

4. Plot results:
```bash
python plot_results.py --exp-ids <experiment_ids> --datasets <dataset_names>
```

### Configuration

The experiments are configured using YAML configuration files:
- `config.yaml` for Beat the Pivot experiments
- `config_baseline.yaml` for baseline experiments

Key configuration parameters include:
- Number of items
- Number of groups
- Epsilon and delta values for fairness constraints
- Dataset-specific configurations
- Protected attributes

## Testing

The project includes a comprehensive test suite to ensure code quality and correctness. To run the tests:

1. Install test dependencies:
```bash
pip install -r tests/requirements-test.txt
```

2. Run all tests:
```bash
pytest tests/
```

3. Run tests with coverage report:
```bash
pytest --cov=active_fair_ranking tests/
```

The test suite includes:
- Unit tests for data structures
- Tests for the Beat the Pivot algorithm
- Tests for metrics and evaluation functions
- Edge case testing
- Configuration testing

## Datasets

The project supports multiple datasets including:
- COMPAS dataset (with sex and race as protected attributes)
- German Credit dataset
- Synthetic datasets with random protected attributes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- Sruthi Gorantla (gorantlas@iisc.ac.in)

## Citation

If you use this code in your research, please cite:
```
@misc{active-fair-ranking,
  author = {Sruthi Gorantla and Sara Ahmadian},
  title = {Fair Active Ranking from Pairwise Preerences},
  year = {2024},
  publisher = {arXiv},
  url = {https://arxiv.org/abs/2402.03252}
}
```
