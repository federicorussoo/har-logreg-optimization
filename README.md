# Gradient Descent vs Block Coordinate Descent for Multiclass Logistic Regression

Comparison of three first-order optimization methods on a multiclass logistic regression problem:

- **GD**: Gradient Descent (fixed stepsize or Armijo rule)
- **Randomized BCGD**: Block Coordinate Gradient Descent, random choice of the block
- **Gauss-Southwell BCGD**: Block Coordinate Gradient Descent, block with the largest gradient

The algorithms are first tested on synthetic data and then compared on a real dataset
(Human Activity Recognition from smartphone sensors), in terms of accuracy, iterations and CPU time.
The full analysis is in the report `Dosvaldi_Pilan_Russo.pdf`.

## Files

Each script is self-contained (it defines the algorithms it uses) and is run on its own.

| File | What it does |
|---|---|
| `01_synthetic_check.py` | Runs the algorithms on synthetic data (report, Section 5.1). Needs no dataset. |
| `02_har_table.py` | Runs the algorithms on the real data and prints iterations, CPU time and accuracy (report, Table 1). |
| `03_fig_time.py` | Plots accuracy against CPU time (report, Figures 1 and 2). |
| `04_fig_other.py` | All other plots: accuracy against iterations and tolerance, most updated features, PCA and prediction errors (report, Figures 3-11). |

## Data

The dataset is **not included** in this repository. Download `train.csv` and `test.csv` from:

> [1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, and Jorge L. Reyes-Ortiz.
> Human activity recognition using smartphones. UCI Machine Learning Repository, 2012.
> Available at Kaggle: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones

and place them in the same folder as the scripts.

## Usage

```bash
pip install numpy scipy pandas scikit-learn matplotlib
python 01_synthetic_check.py
python 02_har_table.py
python 03_fig_time.py
python 04_fig_other.py
```

Scripts 02-04 need `train.csv` and `test.csv`.

## Notes

- All runs use a fixed random seed (42), so iterations and accuracies are reproducible; CPU times depend on the machine.
- In `01_synthetic_check.py` and `02_har_table.py`, `use_armijo` is set to `False`, so the Armijo rule is not run there. Set it to `True` to run it. The Armijo curve in Figure 1 is produced by `03_fig_time.py`.
- In `03_fig_time.py`, the labels on Figure 1 (`94.06% at 0.20s`, `94.3% at 3.84s`) are written by hand from the tolerance-500 runs of `02_har_table.py` (fixed stepsize and Armijo).
- CPU time in `02_har_table.py` includes the one-off setup (e.g. the spectral norm used by GD); in `03_fig_time.py` the time axis starts after it.
