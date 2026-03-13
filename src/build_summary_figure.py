"""Build a standalone summary figure for the README and repo preview."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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


def overall_adjusted_effect(df: pd.DataFrame) -> float:
    """Compute the overall device-adjusted treatment effect in percentage points."""

    weights = df.loc[df["group"] == "control", "device"].value_counts(normalize=True)
    rates = df.groupby(["device", "group"])["converted"].mean().unstack()
    return float(((rates["treatment"] - rates["control"]) * weights).sum() * 100)


def peeking_days(df: pd.DataFrame) -> tuple[int, int]:
    """Return the first significant day and the first later reversion above 0.05."""

    daily_checks = []
    for day in range(3, 29):
        subset = df[df["day_enrolled"] <= day]
        counts = subset.groupby("group")["converted"].sum().reindex(["control", "treatment"])
        nobs = subset.groupby("group").size().reindex(["control", "treatment"])
        p_value = float(proportions_ztest(counts, nobs)[1])
        daily_checks.append((day, p_value))

    first_sig_day = next(day for day, p_value in daily_checks if p_value < 0.05)
    reversion_day = next(day for day, p_value in daily_checks if day > first_sig_day and p_value > 0.05)
    return first_sig_day, reversion_day


def build_figure(df: pd.DataFrame) -> go.Figure:
    """Create a simplified summary figure that reads well in the README."""

    weekly = adjusted_weekly_effects(df)
    counts = df.groupby("group")["converted"].sum().reindex(["control", "treatment"])
    nobs = df.groupby("group").size().reindex(["control", "treatment"])
    overall_p = float(proportions_ztest(counts, nobs)[1])
    topline_lift = float((counts["treatment"] / nobs["treatment"] - counts["control"] / nobs["control"]) * 100)
    adjusted_lift = overall_adjusted_effect(df)
    treatment_share = float(nobs["treatment"] / nobs.sum() * 100)
    control_share = float(nobs["control"] / nobs.sum() * 100)
    first_sig_day, reversion_day = peeking_days(df)

    new_users = df[df["user_type"] == "new"]
    new_counts = new_users.groupby("group")["converted"].sum().reindex(["control", "treatment"])
    new_nobs = new_users.groupby("group").size().reindex(["control", "treatment"])
    new_p = float(proportions_ztest(new_counts, new_nobs)[1])

    returning_users = df[df["user_type"] == "returning"]
    returning_counts = returning_users.groupby("group")["converted"].sum().reindex(["control", "treatment"])
    returning_nobs = returning_users.groupby("group").size().reindex(["control", "treatment"])
    returning_p = float(proportions_ztest(returning_counts, returning_nobs)[1])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weekly["week"],
            y=weekly["adjusted_effect_pp"],
            mode="lines+markers",
            line={"color": TREATMENT_COLOR, "width": 4},
            marker={"size": 10},
        ),
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_annotation(
        x=0.00,
        y=1.10,
        xref="paper",
        yref="paper",
        text=f"<b>Naive topline lift</b><br><span style='font-size:28px'>{topline_lift:.2f} pp</span>",
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.92)",
        bordercolor=CONTROL_COLOR,
        borderwidth=1,
        font={"size": 16},
    )
    fig.add_annotation(
        x=0.50,
        y=1.10,
        xref="paper",
        yref="paper",
        text=f"<b>After device adjustment</b><br><span style='font-size:28px'>{adjusted_lift:.2f} pp</span>",
        showarrow=False,
        align="center",
        bgcolor="rgba(255, 255, 255, 0.92)",
        bordercolor=TREATMENT_COLOR,
        borderwidth=1,
        font={"size": 16},
    )
    fig.add_annotation(
        x=1.00,
        y=1.10,
        xref="paper",
        yref="paper",
        text=(
            "<b>Observed split</b><br>"
            f"<span style='font-size:28px'>{treatment_share:.1f}% / {control_share:.1f}%</span>"
        ),
        showarrow=False,
        align="right",
        bgcolor="rgba(255, 255, 255, 0.92)",
        bordercolor="#A0A0A0",
        borderwidth=1,
        font={"size": 16},
    )
    fig.add_annotation(
        x=0.00,
        y=0.98,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Peeking flips:</b> p < 0.05 by day {first_sig_day}, "
            f"back above 0.05 on day {reversion_day}"
        ),
        showarrow=False,
        align="left",
        bgcolor="rgba(99, 110, 250, 0.10)",
        bordercolor=CONTROL_COLOR,
        borderwidth=1,
        font={"size": 15},
    )
    fig.add_annotation(
        x=1.00,
        y=0.98,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Subgroups stay noisy:</b> new users p = {new_p:.3f}, "
            f"returning users p = {returning_p:.3f}"
        ),
        showarrow=False,
        align="right",
        bgcolor="rgba(239, 85, 59, 0.10)",
        bordercolor=TREATMENT_COLOR,
        borderwidth=1,
        font={"size": 15},
    )

    fig.update_layout(
        title=(
            "Why the Topline Lied"
            f"<br><sup>The checkout redesign looks positive at first, but the adjusted effect is smaller and unstable "
            f"(overall conversion p = {overall_p:.3f}).</sup>"
        ),
        height=760,
        template="plotly_white",
        margin={"t": 220, "r": 60, "b": 60, "l": 60},
        showlegend=False,
    )
    fig.update_xaxes(title_text="Enrollment week", tickfont={"size": 15}, title_font={"size": 16})
    fig.update_yaxes(title_text="Adjusted lift (pp)", showgrid=False, tickfont={"size": 15}, title_font={"size": 16})
    return fig


def main() -> None:
    """Write the summary figure as HTML and PNG artifacts."""

    project_root = Path(__file__).resolve().parents[1]
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(project_root)
    fig = build_figure(df)
    fig.write_html(reports_dir / "experiment_summary.html", include_plotlyjs="cdn")
    fig.write_image(reports_dir / "experiment_summary.png", width=1600, height=900, scale=2)


if __name__ == "__main__":
    main()
