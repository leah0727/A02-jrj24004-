# A02-jrj24004-
our live attempt to do the ping pong assignment

# A02 — California Housing Regression with MLPRegressor

This project trains a neural network regression model using the California Housing dataset from `scikit-learn`.

The workflow includes:

- Loading and preprocessing the dataset
- Train/test split
- Feature scaling using `StandardScaler`
- Training an `MLPRegressor`
- Using `early_stopping=True`
- Evaluating model performance with:
  - R² Score
  - Mean Absolute Error (MAE)
- Generating prediction plots for:
  - Train dataset
  - Test dataset

---

## Project Structure

```
A02-jrj24004-/
│
├── figures/
│   ├── med_house_value_boxplot.png
│   ├── train_actual_vs_pred.png
│   └── test_actual_vs_pred.png
│
├── src/
│   └── ds_pipeline.py
│
├── requirements.txt
└── README.md
```

---

## How to Run

1. Install required packages

```bash
pip install -r requirements.txt
```

2. Run the training pipeline

```bash
python src/ds_pipeline.py
```

---

## Model Details

The project uses `MLPRegressor` from `scikit-learn`.

Custom hyperparameters used:

```python
MLPRegressor(
    random_state=42,
    hidden_layer_sizes=(10, 5),
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=300,
    batch_size=1000,
    activation="relu",
    validation_fraction=0.2,
    early_stopping=True
)
```

---

## Output Files

The pipeline automatically generates and saves:

- `figures/train_actual_vs_pred.png`
- `figures/test_actual_vs_pred.png`
- `figures/med_house_value_boxplot.png`

---

## Collaboration Workflow

| NAME# | Description | Status |
|------|-------------|--------|
| #1 | Data loading | Merged |
| #2 | Added boxplot generation and requirements.txt | Merged |
| PR #1 | Added train/test split | Merged |
| PR #2 | Added MLPRegressor with early stopping, predictions, and evaluation | Merged |
| PR #3 | Added train/test prediction plots | Merged |
| PR #4 | Ran full pipeline and generated plots | Merged |

### GitHub Workflow Evidence

- Multiple branches created
- Pull Requests (PRs) used for each task
- PR comments/reviews included
- PRs merged into main branch
- Branches deleted after merge

---

## Team Members

- Yeongeun Ra (worked individually)
