"""Build a standalone summary figure for the README and repo preview."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.stats.proportion import proportions_ztest


CONTROL_COLOR = "#636EFA"
TREATMENT_COLOR = "#EF553B"


def load_data(project_root: Path) -> pd.DataFrame:
    """Load the simulated experiment export."""

    data_path = project_root / "data" / "ab_test_raw.csv"
    df = pd.read_csv(data_path)
    df["week"] = pd.cut(
        df["day_enrolled"],
        bins=[0, 7, 14, 21, 28],
        labels=["Week 1", "Week 2", "Week 3", "Week 4"],
        include_lowest=True,
    )
    return df


def adjusted_weekly_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Compute device-adjusted weekly treatment effects."""

    rows = []
    for week_label, subset in df.groupby("week", observed=False):
        weights = subset.loc[subset["group"] == "control", "device"].value_counts(normalize=True)
        rates = subset.groupby(["device", "group"])["converted"].mean().unstack()
        effect = float(((rates["treatment"] - rates["control"]) * weights).sum() * 100)
        rows.append({"week": week_label, "adjusted_effect_pp": effect})
    return pd.DataFrame(rows)


def subgroup_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Compute point estimates and confidence intervals for key subgroups."""

    def summarize(label: str, subset: pd.DataFrame) -> dict[str, float | str]:
        grouped = subset.groupby("group")["converted"].agg(["sum", "count", "mean"]).reindex(["control", "treatment"])
        p_t = grouped.loc["treatment", "mean"]
        p_c = grouped.loc["control", "mean"]
        n_t = grouped.loc["treatment", "count"]
        n_c = grouped.loc["control", "count"]
        diff = p_t - p_c
        variance = (p_t * (1 - p_t) / n_t) + (p_c * (1 - p_c) / n_c)
        margin = 1.96 * variance**0.5
        return {
            "segment": label,
            "effect_pp": diff * 100,
            "ci_low_pp": (diff - margin) * 100,
            "ci_high_pp": (diff + margin) * 100,
        }

    rows = [summarize("Overall", df)]
    rows.append(summarize("New users", df[df["user_type"] == "new"]))
    rows.append(summarize("Returning users", df[df["user_type"] == "returning"]))
    rows.append(summarize("Mobile", df[df["device"] == "mobile"]))
    rows.append(summarize("Desktop", df[df["device"] == "desktop"]))
    rows.append(summarize("Tablet", df[df["device"] == "tablet"]))
    return pd.DataFrame(rows)


def build_figure(df: pd.DataFrame) -> go.Figure:
    """Create the stakeholder-ready summary figure."""

    weekly = adjusted_weekly_effects(df)
    conversion_by_device = (
        df.groupby(["device", "group"])["converted"]
        .mean()
        .mul(100)
        .reset_index(name="conversion_rate_pct")
    )
    forest = subgroup_effects(df).iloc[::-1]

    overall = df.groupby("group")["converted"].mean().mul(100)
    counts = df.groupby("group")["converted"].sum().reindex(["control", "treatment"])
    nobs = df.groupby("group").size().reindex(["control", "treatment"])
    overall_p = float(proportions_ztest(counts, nobs)[1])

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Device-adjusted treatment effect by week",
            "Conversion rate by device",
            "Subgroup treatment effects",
        ),
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(
            x=weekly["week"],
            y=weekly["adjusted_effect_pp"],
            mode="lines+markers",
            line={"color": TREATMENT_COLOR, "width": 3},
            name="Adjusted lift",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=conversion_by_device[conversion_by_device["group"] == "control"]["device"],
            y=conversion_by_device[conversion_by_device["group"] == "control"]["conversion_rate_pct"],
            marker_color=CONTROL_COLOR,
            name="Control",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=conversion_by_device[conversion_by_device["group"] == "treatment"]["device"],
            y=conversion_by_device[conversion_by_device["group"] == "treatment"]["conversion_rate_pct"],
            marker_color=TREATMENT_COLOR,
            name="Treatment",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=forest["effect_pp"],
            y=forest["segment"],
            mode="markers",
            marker={"color": TREATMENT_COLOR, "size": 9},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": forest["ci_high_pp"] - forest["effect_pp"],
                "arrayminus": forest["effect_pp"] - forest["ci_low_pp"],
            },
            name="Point estimate",
        ),
        row=3,
        col=1,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=float(overall["control"]), line_color=CONTROL_COLOR, line_dash="dash", row=2, col=1)
    fig.add_hline(y=float(overall["treatment"]), line_color=TREATMENT_COLOR, line_dash="dash", row=2, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=3, col=1)

    fig.update_layout(
        title=f"Experiment Summary: Why the Topline Lied (overall conversion p = {overall_p:.3f})",
        height=1100,
        barmode="group",
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Enrollment week", row=1, col=1)
    fig.update_yaxes(title_text="Lift (pp)", row=1, col=1, showgrid=False)
    fig.update_xaxes(title_text="Device", row=2, col=1)
    fig.update_yaxes(title_text="Conversion rate (%)", row=2, col=1, showgrid=False)
    fig.update_xaxes(title_text="Lift (pp)", row=3, col=1)
    fig.update_yaxes(title_text="", row=3, col=1, showgrid=False)
    return fig


def main() -> None:
    """Write the summary figure as a standalone HTML artifact."""

    project_root = Path(__file__).resolve().parents[1]
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(project_root)
    fig = build_figure(df)
    fig.write_html(reports_dir / "experiment_summary.html", include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
