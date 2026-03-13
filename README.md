# When A/B Tests Lie: A Case Study in Experiment Validation

This project is a portfolio-style analytics case study built around a deliberately flawed e-commerce experiment. I simulate a 50,000-user checkout A/B test, walk through the confident-but-naive topline readout, and then show how a proper experiment audit changes the recommendation. The focus is not just on coding the analysis, but on demonstrating judgment: power analysis, validation checks, causal skepticism, and stakeholder-ready communication.

## At a glance

- Naive topline result: treatment conversion is higher by **0.64 percentage points** (`p = 0.028`), and revenue per user is also significant (`p = 0.016`).
- Audit result: once I standardize for device mix, the lift shrinks to about **0.42 percentage points**.
- Decision quality check: the naive lift is still below the notebook's pre-test minimum detectable effect, so even the optimistic read is not a practically strong win.
- Randomization issue: treatment receives materially more desktop traffic (`36.33%` vs `33.67%` in control), driven by returning desktop users being over-assigned.
- Timing issue: the cumulative p-value first crosses below `0.05` on day `3`, rises back above it on day `19`, then ends significant again, which is exactly the kind of unstable read that invites premature shipping.
- Subgroup caution: both `new` and `returning` user slices are non-significant in the tuned run, so the case study resists the usual overconfident “it works for this segment” story.

## Preview

A standalone exported summary figure is available at [reports/experiment_summary.html](/home/yakvrz/Projects/ab-tests-lie/reports/experiment_summary.html).

## Key findings

- The topline result looks significant, but the randomization is compromised by a sample ratio mismatch.
- Treatment inherited more returning desktop users, creating a Simpson's paradox effect that inflates pooled conversion.
- The device-adjusted treatment effect spikes early, weakens sharply through the middle of the run, and ends far below the topline lift.
- Subgroup point estimates are easy to over-read; once confidence intervals are shown, several slices are too noisy to support strong product decisions.

## Methodology

- Simulated user-level experiment data with known ground truth and a seeded randomization bug
- Pre-experiment power analysis to estimate the minimum detectable effect
- SRM detection with group-level and segment-level assignment audits
- Stratified analysis to diagnose Simpson's paradox and device-mix inflation
- Sequential read simulation to illustrate the risk of early peeking
- Week-by-week effect decomposition to isolate novelty decay
- Subgroup effect estimation, confidence intervals, and interaction testing

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/simulate.py
.venv/bin/python src/build_notebook.py
.venv/bin/python src/build_summary_figure.py
.venv/bin/jupyter notebook notebooks/analysis.ipynb
```

If you prefer to verify the notebook end-to-end in the terminal first, run:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

## Repo contents

```text
ab-tests-lie/
├── README.md
├── requirements.txt
├── data/
│   ├── ab_test_raw.csv
│   └── ground_truth.json
├── notebooks/
│   └── analysis.ipynb
├── reports/
│   └── experiment_summary.html
└── src/
    ├── build_summary_figure.py
    ├── build_notebook.py
    └── simulate.py
```

## Author

Koby Raz  
[GitHub](https://github.com/yakvrz)
