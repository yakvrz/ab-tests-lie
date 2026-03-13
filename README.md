# When A/B Tests Lie: A Case Study in Experiment Validation

This repository is about a simple idea: an A/B test can be statistically significant and still point a team toward the wrong product decision. The case study uses a deliberately flawed e-commerce checkout experiment to show how that happens in practice, and how a careful audit can separate a real improvement from a misleading win. The emphasis is on the judgment that makes experiment results trustworthy: power analysis, validation checks, causal skepticism, and clear stakeholder communication.

Bottom line: the checkout redesign looks like a win at the topline, but the experiment is compromised and should not be used to justify shipping.

## At a glance

- Naive topline: treatment conversion is higher by **0.64 percentage points** (`p = 0.028`), and revenue per user is also significant (`p = 0.016`).
- After adjusting for device mix, the lift shrinks to about **0.42 percentage points**.
- Even the naive lift is below the notebook's pre-test minimum detectable effect, meaning the optimistic read is not a practically strong win.
- Treatment gets materially more desktop traffic (`36.33%` vs `33.67%` in control), driven by returning desktop users being over-assigned.
- The cumulative p-value first drops below `0.05` on day `3`, rises back above it on day `19`, then ends significant again, showing how unstable early reads can be.
- Both `new` and `returning` user slices are non-significant in the tuned run, which blocks the usual overconfident subgroup story.

## Preview

A standalone summary figure is available at [reports/experiment_summary.html](/home/yakvrz/Projects/ab-tests-lie/reports/experiment_summary.html).

## Key findings

- The topline result looks significant, but the randomization is compromised by a sample ratio mismatch (SRM).
- Treatment inherited more returning desktop users, creating a Simpson's paradox effect that inflates pooled conversion.
- The device-adjusted treatment effect spikes early, weakens sharply through the middle of the run, and ends far below the topline lift.
- Subgroup point estimates are easy to over-read; once confidence intervals are shown, the slices are too noisy to support strong product decisions.

## Methodology

- Simulated user-level experiment data with known ground truth and a seeded randomization bug
- Pre-experiment power analysis to estimate the minimum detectable effect (MDE)
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
.venv/bin/jupyter lab notebooks/analysis.ipynb
```

To verify the notebook end-to-end in the terminal first, run:

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
