# When A/B Tests Lie: A Case Study in Experiment Validation

This repository is about a simple idea: an A/B test can be statistically significant and still point a team toward the wrong product decision. The case study uses a simulated e-commerce checkout experiment designed as a composite of realistic experimentation failure modes, showing how a careful audit can separate a real improvement from a misleading win. The emphasis is on the judgment that makes experiment results trustworthy: power analysis, validation checks, causal skepticism, and clear stakeholder communication.

Bottom line: the checkout redesign looks like a win at the topline, but the experiment is compromised and should not be used to justify shipping.

## At a glance

- Scenario: CartCo tests a redesigned checkout flow against the existing version on 50,000 users.
- First impression: the redesign appears to win. Conversion is higher by **0.64 percentage points** (`p = 0.028`), and revenue per user is also higher (`p = 0.016`).
- Deeper audit: the result is inflated by a randomization problem, distorted by traffic mix, and unstable over time.
- Recommendation: this test result is not reliable enough to justify shipping the redesign.
- The notebook contains the full walkthrough, including the topline read, the audit checks, and the final decision.

## Methodology

- Simulated user-level experiment data with known ground truth and a seeded randomization bug
- Pre-experiment power analysis to estimate the minimum detectable effect (MDE)
- SRM detection with group-level and segment-level assignment audits
- Stratified analysis to diagnose Simpson's paradox and device-mix inflation
- Sequential read simulation to illustrate the risk of early peeking
- Week-by-week effect decomposition to evaluate the effect trajectory over time
- Subgroup effect estimation, confidence intervals, and interaction testing

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/simulate.py      # generate the simulated experiment data
.venv/bin/python src/build_notebook.py  # rebuild the notebook from the source template
.venv/bin/jupyter lab notebooks/analysis.ipynb
```

To verify the notebook end-to-end in the terminal first, run:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

## Author

Koby Raz  
[GitHub](https://github.com/yakvrz)
