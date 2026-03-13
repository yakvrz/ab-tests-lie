"""Build the CartCo analysis notebook from reusable cell templates."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def md(text: str):
    """Create a markdown notebook cell."""

    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    """Create a code notebook cell."""

    return nbf.v4.new_code_cell(dedent(text).strip())


def build_notebook() -> nbf.NotebookNode:
    """Assemble the full case-study notebook."""

    cells = [
        md(
            """
            # When A/B Tests Lie

            **CartCo experiment review**

            I simulated a realistic 4-week experiment where the topline says "ship it" and the audit says "not so fast." The goal of this notebook is to show the workflow I use before I trust any experiment readout: size the test, inspect the topline, then pressure-test the randomization, composition, timing, and subgroup story before making a product recommendation.
            """
        ),
        md(
            """
            ## Part 1. The Setup

            CartCo redesigned its checkout flow to reduce friction at the last step before purchase. The product hypothesis was simple: a cleaner flow should lift checkout conversion.

            The experiment was planned as a 50/50 split over 4 weeks with about 50K users. Before looking at outcomes, I want to ground the conversation in power: with this traffic and an overall baseline near 12%, we should only expect to reliably detect changes on the order of roughly one percentage point or more.
            """
        ),
        md(
            """
            ### Load the experiment data

            I start by loading the raw simulated export and the ground-truth configuration that I will only use in the appendix. Then I add a clean week label so every later section can reuse the same time buckets.
            """
        ),
        code(
            """
            import json
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
            import plotly.io as pio
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            from IPython.display import Markdown, display
            from plotly.subplots import make_subplots
            from scipy import stats
            from scipy.optimize import brentq
            from scipy.stats import chisquare
            from statsmodels.stats.power import NormalIndPower
            from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

            pio.templates.default = "plotly_white"
            pd.options.display.float_format = "{:,.4f}".format

            CONTROL_COLOR = "#636EFA"
            TREATMENT_COLOR = "#EF553B"

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "data").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent
            """
        ),
        code(
            """
            data_path = PROJECT_ROOT / "data" / "ab_test_raw.csv"
            truth_path = PROJECT_ROOT / "data" / "ground_truth.json"

            df = pd.read_csv(data_path, parse_dates=["signup_date"])
            with truth_path.open() as handle:
                ground_truth = json.load(handle)

            df["week"] = pd.cut(
                df["day_enrolled"],
                bins=[0, 7, 14, 21, 28],
                labels=["Week 1", "Week 2", "Week 3", "Week 4"],
                include_lowest=True,
            )
            """
        ),
        md(
            """
            ### Build a few reusable helpers

            I keep the helper functions lightweight and explicit so the notebook stays easy to audit. These functions handle the proportion math, device-standardized estimates, and repeated formatting used across the sections below.
            """
        ),
        code(
            """
            def proportion_diff_ci(success_t, total_t, success_c, total_c, z_value=1.96):
                p_t = success_t / total_t
                p_c = success_c / total_c
                diff = p_t - p_c
                variance = (p_t * (1 - p_t) / total_t) + (p_c * (1 - p_c) / total_c)
                margin = z_value * np.sqrt(variance)
                return diff, diff - margin, diff + margin


            def two_prop_summary(frame):
                summary = (
                    frame.groupby("group")["converted"]
                    .agg(conversions="sum", users="count", conversion_rate="mean")
                    .reindex(["control", "treatment"])
                )
                stat, p_value = proportions_ztest(summary["conversions"], summary["users"])
                diff, ci_low, ci_high = proportion_diff_ci(
                    summary.loc["treatment", "conversions"],
                    summary.loc["treatment", "users"],
                    summary.loc["control", "conversions"],
                    summary.loc["control", "users"],
                )
                return summary, stat, p_value, diff, ci_low, ci_high
            """
        ),
        code(
            """
            def adjusted_effect(frame, segment_col="device", z_value=1.96):
                weights = (
                    frame.loc[frame["group"] == "control", segment_col]
                    .value_counts(normalize=True)
                    .sort_index()
                )
                segment_stats = (
                    frame.groupby([segment_col, "group"])["converted"]
                    .agg(["sum", "count", "mean"])
                    .reset_index()
                )
                effects = []
                variances = []
                for segment, weight in weights.items():
                    subset = segment_stats[segment_stats[segment_col] == segment].set_index("group")
                    p_t = subset.loc["treatment", "mean"]
                    p_c = subset.loc["control", "mean"]
                    n_t = subset.loc["treatment", "count"]
                    n_c = subset.loc["control", "count"]
                    effects.append(weight * (p_t - p_c))
                    variances.append((weight ** 2) * ((p_t * (1 - p_t) / n_t) + (p_c * (1 - p_c) / n_c)))
                effect = float(np.sum(effects))
                margin = z_value * float(np.sqrt(np.sum(variances)))
                return effect, effect - margin, effect + margin


            def sample_size_for_one_point_lift(baseline_rate, alpha=0.05, power=0.80):
                target_rate = min(0.999, baseline_rate + 0.01)
                effect_size = abs(proportion_effectsize(baseline_rate, target_rate))
                return int(np.ceil(NormalIndPower().solve_power(effect_size, alpha=alpha, power=power, ratio=1.0)))
            """
        ),
        code(
            """
            def format_percent(series):
                return series.mul(100).round(2).astype(str) + "%"


            def style_table(frame, formats):
                return frame.style.format(formats).hide(axis="index")


            def subgroup_effect(label, frame):
                summary, stat, p_value, diff, ci_low, ci_high = two_prop_summary(frame)
                return {
                    "segment": label,
                    "effect_pp": diff * 100,
                    "ci_low_pp": ci_low * 100,
                    "ci_high_pp": ci_high * 100,
                    "p_value": p_value,
                    "control_rate": summary.loc["control", "conversion_rate"] * 100,
                    "treatment_rate": summary.loc["treatment", "conversion_rate"] * 100,
                    "users": int(summary["users"].sum()),
                }
            """
        ),
        md(
            """
            ### Take a first look at the raw file

            Before running any inference, I sanity-check the table shape, the schema, and the basic composition of the sample. This is the fastest way to catch malformed exports or obviously strange traffic patterns.
            """
        ),
        code(
            """
            print(f"Shape: {df.shape}")
            display(df.head())
            display(df.dtypes.to_frame("dtype"))
            """
        ),
        code(
            """
            group_counts = (
                df["group"].value_counts()
                .rename_axis("group")
                .reset_index(name="users")
                .assign(share=lambda frame: frame["users"] / frame["users"].sum())
            )

            device_split = (
                df["device"].value_counts(normalize=True)
                .rename_axis("device")
                .reset_index(name="share")
            )

            user_type_split = (
                df["user_type"].value_counts(normalize=True)
                .rename_axis("user_type")
                .reset_index(name="share")
            )

            display(style_table(group_counts, {"users": "{:,.0f}", "share": "{:.2%}"}))
            display(style_table(device_split, {"share": "{:.2%}"}))
            display(style_table(user_type_split, {"share": "{:.2%}"}))
            """
        ),
        md(
            """
            ### Estimate the minimum detectable effect

            This is the expectation-setting step I like to do before looking at results. If the sample can only reliably detect a 1 to 1.5 percentage point lift, then any tiny late-stage wiggle should be treated as noise rather than a product win.
            """
        ),
        code(
            """
            group_sizes = df["group"].value_counts()
            control_rate = df.loc[df["group"] == "control", "converted"].mean()
            ratio = group_sizes["treatment"] / group_sizes["control"]

            power_solver = NormalIndPower()
            effect_size = power_solver.solve_power(
                effect_size=None,
                nobs1=group_sizes["control"],
                alpha=0.05,
                power=0.80,
                ratio=ratio,
            )

            mde_rate = brentq(
                lambda candidate: abs(proportion_effectsize(control_rate, candidate)) - effect_size,
                control_rate + 1e-6,
                0.50,
            )
            mde_pp = (mde_rate - control_rate) * 100

            display(
                Markdown(
                    f"With a control baseline of **{control_rate:.2%}** and roughly "
                    f"**{group_sizes['control']:,} / {group_sizes['treatment']:,}** users per arm, "
                    f"the 80% power MDE is about **{mde_pp:.2f} percentage points**."
                )
            )
            """
        ),
        md(
            """
            ### Plot enrollment over time

            I want to know whether traffic came in steadily or front-loaded. The day-level volume matters later when I look at novelty and when I simulate what an impatient analyst would have seen if they peeked early.
            """
        ),
        code(
            """
            enrollment_by_day = (
                df.groupby(["day_enrolled", "group"])
                .size()
                .reset_index(name="users")
            )

            fig = px.bar(
                enrollment_by_day,
                x="day_enrolled",
                y="users",
                color="group",
                barmode="group",
                color_discrete_map={"control": CONTROL_COLOR, "treatment": TREATMENT_COLOR},
                title="Enrollment Volume by Day",
                labels={"day_enrolled": "Experiment day", "users": "Users enrolled", "group": "Group"},
            )
            fig.update_layout(font={"size": 12})
            fig.update_yaxes(showgrid=False)
            fig.show()
            """
        ),
        md(
            """
            ## Part 2. The Naive Analysis

            If I stop at the topline, this experiment looks strong. The treatment arm converted more, generated more revenue per user, and passes a standard significance test. This is the part of the story that gets people excited fast.
            """
        ),
        md(
            """
            ### Summarize the topline conversion rates

            I start with the table most dashboards lead with: total users, total conversions, and conversion rate by arm. This is the clean version of the story before I inspect whether the groups are actually comparable.
            """
        ),
        code(
            """
            overall_summary, overall_stat, overall_p, overall_diff, overall_ci_low, overall_ci_high = two_prop_summary(df)

            topline_table = overall_summary.assign(
                conversion_rate_pct=lambda frame: frame["conversion_rate"] * 100
            ).reset_index().rename(columns={"group": "Group"})

            display(
                style_table(
                    topline_table[["Group", "users", "conversions", "conversion_rate_pct"]],
                    {"users": "{:,.0f}", "conversions": "{:,.0f}", "conversion_rate_pct": "{:.2f}%"},
                )
            )
            """
        ),
        md(
            """
            ### Test the primary and secondary metrics

            Now I quantify the headline result. For conversion I use a two-proportion z-test and a 95% confidence interval on the lift. For revenue per user, I compare mean revenue including zeros because that is the stakeholder-facing business metric.
            """
        ),
        code(
            """
            revenue_summary = (
                df.groupby("group")["revenue"]
                .agg(mean_revenue_per_user="mean", std="std", users="count")
                .reindex(["control", "treatment"])
            )
            revenue_test = stats.ttest_ind(
                df.loc[df["group"] == "treatment", "revenue"],
                df.loc[df["group"] == "control", "revenue"],
                equal_var=False,
            )

            test_results = pd.DataFrame(
                {
                    "metric": ["Conversion rate", "Revenue per user"],
                    "estimate": [overall_diff * 100, revenue_summary.loc["treatment", "mean_revenue_per_user"] - revenue_summary.loc["control", "mean_revenue_per_user"]],
                    "ci_low": [overall_ci_low * 100, np.nan],
                    "ci_high": [overall_ci_high * 100, np.nan],
                    "test_stat": [overall_stat, revenue_test.statistic],
                    "p_value": [overall_p, revenue_test.pvalue],
                }
            )

            display(style_table(test_results, {"estimate": "{:.3f}", "ci_low": "{:.3f}", "ci_high": "{:.3f}", "test_stat": "{:.3f}", "p_value": "{:.5f}"}))
            """
        ),
        code(
            """
            weekly_checkout_starts = len(df) / 4
            annual_uplift = (
                revenue_summary.loc["treatment", "mean_revenue_per_user"]
                - revenue_summary.loc["control", "mean_revenue_per_user"]
            ) * weekly_checkout_starts * 52

            display(
                Markdown(
                    "### Naive conclusion\\n"
                    f"The new checkout flow increased conversion by **{overall_diff * 100:.2f} percentage points** "
                    f"(95% CI: **{overall_ci_low * 100:.2f} to {overall_ci_high * 100:.2f} pp**, "
                    f"p = **{overall_p:.5f}**).\\n\\n"
                    f"Treatment also lifted revenue per user by **${revenue_summary.loc['treatment', 'mean_revenue_per_user'] - revenue_summary.loc['control', 'mean_revenue_per_user']:.2f}**, "
                    f"which annualizes to roughly **${annual_uplift:,.0f}** if this traffic level holds. "
                    "If I stopped here, I would recommend shipping."
                )
            )
            """
        ),
        md(
            """
            ### Visualize the topline result

            This is the slide-deck version of the experiment: two bars, two clean confidence intervals, and one apparently easy decision.
            """
        ),
        code(
            """
            naive_chart = topline_table.copy()
            naive_chart["ci_low"] = [np.nan, overall_ci_low * 100]
            naive_chart["ci_high"] = [np.nan, overall_ci_high * 100]

            fig = go.Figure()
            for group_name, color in [("control", CONTROL_COLOR), ("treatment", TREATMENT_COLOR)]:
                row = naive_chart[naive_chart["Group"] == group_name]
                error = None
                if group_name == "treatment":
                    error = dict(
                        type="data",
                        symmetric=False,
                        array=[row["ci_high"].iloc[0] - row["conversion_rate_pct"].iloc[0]],
                        arrayminus=[row["conversion_rate_pct"].iloc[0] - row["ci_low"].iloc[0]],
                    )
                fig.add_bar(x=[group_name.title()], y=row["conversion_rate_pct"], marker_color=color, error_y=error)

            fig.update_layout(title="Naive Topline: Treatment Looks Like a Clear Winner", yaxis_title="Conversion rate (%)", xaxis_title="")
            fig.update_yaxes(showgrid=False)
            fig.show()
            """
        ),
        md(
            """
            ## Part 3. The Audit

            Before I sign off on any experiment, I run the same validation checks. This is the difference between reporting a metric and trusting a metric.
            """
        ),
        md(
            """
            ### Check 1. Sample Ratio Mismatch

            The very first audit question is whether the randomization still looks random. If a 50/50 experiment comes back materially off-balance, I assume something leaked before I assume the treatment worked.
            """
        ),
        code(
            """
            observed_counts = df["group"].value_counts().reindex(["control", "treatment"])
            expected_counts = np.repeat(len(df) / 2, 2)
            srm_stat, srm_p = chisquare(observed_counts, f_exp=expected_counts)

            srm_table = pd.DataFrame(
                {
                    "group": observed_counts.index.str.title(),
                    "users": observed_counts.values,
                    "share": observed_counts.values / observed_counts.sum(),
                }
            )

            display(style_table(srm_table, {"users": "{:,.0f}", "share": "{:.2%}"}))
            display(Markdown(f"SRM chi-squared statistic = **{srm_stat:.2f}**, p-value = **{srm_p:.5f}**."))
            """
        ),
        code(
            """
            segment_assignment = (
                df.groupby(["device", "user_type", "group"])
                .size()
                .reset_index(name="users")
            )
            segment_summary = (
                segment_assignment.pivot(index=["device", "user_type"], columns="group", values="users")
                .fillna(0)
                .reset_index()
                .assign(treatment_share=lambda frame: frame["treatment"] / (frame["control"] + frame["treatment"]))
                .sort_values("treatment_share", ascending=False)
            )

            display(
                style_table(
                    segment_summary,
                    {"control": "{:,.0f}", "treatment": "{:,.0f}", "treatment_share": "{:.2%}"},
                )
            )
            """
        ),
        code(
            """
            ratio_chart = segment_summary.copy()
            ratio_chart["segment"] = ratio_chart["device"].str.title() + " × " + ratio_chart["user_type"].str.title()
            ratio_chart["deviation_pp"] = (ratio_chart["treatment_share"] - 0.50) * 100
            ratio_chart["highlight"] = np.where(ratio_chart["segment"] == "Desktop × Returning", "Anomaly", "Expected")

            fig = px.bar(
                ratio_chart,
                x="segment",
                y="deviation_pp",
                color="highlight",
                color_discrete_map={"Expected": CONTROL_COLOR, "Anomaly": TREATMENT_COLOR},
                title="SRM Audit: Returning Desktop Users Were Over-Assigned to Treatment",
                labels={"deviation_pp": "Treatment share minus 50/50 target (pp)", "segment": "Segment"},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_annotation(x="Desktop × Returning", y=5.55, text="55.6% treatment", showarrow=True, arrowhead=2)
            fig.update_yaxes(showgrid=False)
            fig.show()
            """
        ),
        md(
            """
            SRM means the groups are no longer interchangeable. In plain English: treatment got too many returning desktop users, and desktop users convert better even when nothing changes in checkout. That is enough to inflate the topline before I even get to user behavior.
            """
        ),
        md(
            """
            ### Check 2. Simpson's Paradox

            Once I know the mix is skewed, I re-cut the conversion rate within comparable slices. If treatment only wins because it got more high-intent traffic, the overall lift will shrink as soon as I compare like with like.
            """
        ),
        code(
            """
            conversion_by_device = (
                df.groupby(["device", "group"])["converted"]
                .mean()
                .reset_index(name="conversion_rate")
            )

            adjusted_diff, adjusted_low, adjusted_high = adjusted_effect(df, segment_col="device")
            device_mix = (
                df.groupby("group")["device"]
                .value_counts(normalize=True)
                .rename("share")
                .reset_index()
            )

            display(
                style_table(
                    conversion_by_device.assign(conversion_rate=lambda frame: frame["conversion_rate"] * 100),
                    {"conversion_rate": "{:.2f}%"},
                )
            )
            display(style_table(device_mix, {"share": "{:.2%}"}))
            display(Markdown(f"Device-adjusted treatment lift = **{adjusted_diff * 100:.2f} pp** (95% CI: **{adjusted_low * 100:.2f} to {adjusted_high * 100:.2f} pp**)."))
            """
        ),
        code(
            """
            device_chart = conversion_by_device.copy()
            overall_rates = df.groupby("group")["converted"].mean().mul(100)

            fig = px.bar(
                device_chart.assign(conversion_rate=lambda frame: frame["conversion_rate"] * 100),
                x="device",
                y="conversion_rate",
                color="group",
                barmode="group",
                color_discrete_map={"control": CONTROL_COLOR, "treatment": TREATMENT_COLOR},
                title="Simpson's Paradox: Treatment Looks Great Overall Because It Has More Desktop Users",
                labels={"device": "Device", "conversion_rate": "Conversion rate (%)", "group": "Group"},
            )
            fig.add_hline(y=overall_rates["control"], line_color=CONTROL_COLOR, line_dash="dash")
            fig.add_hline(y=overall_rates["treatment"], line_color=TREATMENT_COLOR, line_dash="dash")
            fig.add_annotation(x="mobile", y=overall_rates["treatment"] + 0.1, text="Overall treatment", showarrow=False, font={"color": TREATMENT_COLOR})
            fig.add_annotation(x="mobile", y=overall_rates["control"] - 0.2, text="Overall control", showarrow=False, font={"color": CONTROL_COLOR})
            fig.update_yaxes(showgrid=False)
            fig.show()
            """
        ),
        md(
            """
            The overall treatment rate is higher, but the adjusted lift is much smaller. That is the telltale pattern of Simpson's paradox: treatment inherits better traffic, so the pooled number flatters the redesign more than the like-for-like comparisons do.
            """
        ),
        md(
            """
            ### Check 3. Peeking Bias

            Next I ask what would have happened if someone checked results every day. Even when the final recommendation is "do not ship," repeated looks can still lock a team into an overconfident story long before the audit is finished.
            """
        ),
        code(
            """
            daily_checks = []
            for day in range(3, 29):
                subset = df[df["day_enrolled"] <= day]
                summary, stat, p_value, diff, ci_low, ci_high = two_prop_summary(subset)
                daily_checks.append(
                    {
                        "day": day,
                        "p_value": p_value,
                        "effect_pp": diff * 100,
                        "ci_low_pp": ci_low * 100,
                        "ci_high_pp": ci_high * 100,
                    }
                )

            daily_checks = pd.DataFrame(daily_checks)
            display(style_table(daily_checks.head(10), {"p_value": "{:.5f}", "effect_pp": "{:.2f}", "ci_low_pp": "{:.2f}", "ci_high_pp": "{:.2f}"}))
            """
        ),
        code(
            """
            first_sig_day = int(daily_checks.loc[daily_checks["p_value"] < 0.05, "day"].min())
            reversion_days = daily_checks.loc[
                (daily_checks["day"] > first_sig_day) & (daily_checks["p_value"] > 0.05),
                "day",
            ]
            reversion_day = int(reversion_days.min()) if not reversion_days.empty else None

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=daily_checks["day"],
                    y=daily_checks["p_value"],
                    mode="lines+markers",
                    line={"color": TREATMENT_COLOR, "width": 3},
                    name="Cumulative p-value",
                )
            )
            fig.add_hline(y=0.05, line_dash="dash", line_color="red")
            fig.add_hrect(y0=0, y1=0.05, fillcolor="rgba(239, 85, 59, 0.10)", line_width=0)
            first_sig_p = float(daily_checks.loc[daily_checks["day"] == first_sig_day, "p_value"].iloc[0])
            final_p = float(daily_checks.loc[daily_checks["day"] == 28, "p_value"].iloc[0])

            fig.add_annotation(x=first_sig_day, y=first_sig_p, text=f"First significant read: day {first_sig_day}", showarrow=True, arrowhead=2)
            if reversion_day is not None:
                reversion_p = float(daily_checks.loc[daily_checks["day"] == reversion_day, "p_value"].iloc[0])
                fig.add_annotation(x=reversion_day, y=reversion_p, text=f"Back above 0.05: day {reversion_day}", showarrow=True, arrowhead=2)
            fig.add_annotation(x=28, y=final_p, text="Final read", showarrow=True, arrowhead=2)
            fig.update_layout(title="Sequential Peeking: The Experiment Looked Convincing Almost Immediately", xaxis_title="Latest day included", yaxis_title="Two-proportion z-test p-value")
            fig.update_yaxes(showgrid=False)
            fig.show()
            """
        ),
        code(
            """
            reversion_text = (
                f"The cumulative p-value first dropped below 0.05 on **day {first_sig_day}**, "
                f"drifted back above the threshold on **day {reversion_day}**, "
                "and then finished significant again at the end of the run."
                if reversion_day is not None
                else f"The cumulative p-value first dropped below 0.05 on **day {first_sig_day}** and stayed below it."
            )

            display(
                Markdown(
                    "This is exactly why I do not like ad hoc peeking. "
                    + reversion_text
                    + " If someone had called the test on the first early win, they would have locked onto a moving target before the randomization audit and the effect-decay work were done."
                )
            )
            """
        ),
        md(
            """
            ### Check 4. Effect Trajectory Over Time

            To estimate the durable effect, I look at the treatment lift over enrollment week after standardizing for device mix. That strips out the composition bug and shows whether the redesign keeps helping once the early excitement fades and the traffic mix settles down.
            """
        ),
        code(
            """
            weekly_rows = []
            for week_label, subset in df.groupby("week", observed=False):
                raw_summary, _, _, raw_diff, _, _ = two_prop_summary(subset)
                adj_diff, adj_low, adj_high = adjusted_effect(subset, segment_col="device")
                weekly_rows.append(
                    {
                        "week": week_label,
                        "raw_effect_pp": raw_diff * 100,
                        "adjusted_effect_pp": adj_diff * 100,
                        "ci_low_pp": adj_low * 100,
                        "ci_high_pp": adj_high * 100,
                        "control_rate_pct": raw_summary.loc["control", "conversion_rate"] * 100,
                        "treatment_rate_pct": raw_summary.loc["treatment", "conversion_rate"] * 100,
                    }
                )

            weekly_effects = pd.DataFrame(weekly_rows)
            display(style_table(weekly_effects, {"raw_effect_pp": "{:.2f}", "adjusted_effect_pp": "{:.2f}", "ci_low_pp": "{:.2f}", "ci_high_pp": "{:.2f}", "control_rate_pct": "{:.2f}", "treatment_rate_pct": "{:.2f}"}))
            """
        ),
        code(
            """
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=weekly_effects["week"],
                    y=weekly_effects["ci_high_pp"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=weekly_effects["week"],
                    y=weekly_effects["ci_low_pp"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(239, 85, 59, 0.15)",
                    name="95% CI",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=weekly_effects["week"],
                    y=weekly_effects["adjusted_effect_pp"],
                    mode="lines+markers",
                    line={"color": TREATMENT_COLOR, "width": 3},
                    name="Adjusted lift",
                )
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(title="Effect Trajectory: The Device-Adjusted Lift Is Strong Early and Weak Later", xaxis_title="Enrollment week", yaxis_title="Treatment lift (pp)")
            fig.update_yaxes(showgrid=False)
            fig.show()
            """
        ),
        code(
            """
            week1_effect = weekly_effects.loc[weekly_effects["week"] == "Week 1", "adjusted_effect_pp"].iloc[0]
            week4_effect = weekly_effects.loc[weekly_effects["week"] == "Week 4", "adjusted_effect_pp"].iloc[0]

            display(
                Markdown(
                    f"This is the core behavior story. The device-adjusted lift starts at **{week1_effect:.2f} pp** in week 1, "
                    "turns negative in the middle of the run, and only recovers to a small effect by the final week "
                    f"(**{week4_effect:.2f} pp**). The important point is not a perfect novelty story; it is that the effect does not settle into a clean, durable win."
                )
            )
            """
        ),
        md(
            """
            ### Check 5. Underpowered subgroup claims

            Subgroup stories are where a lot of experiment decks go off the rails. I still calculate them, but I want the confidence intervals on the page and I want to separate "there is an effect" from "the effect is meaningfully different across segments."
            """
        ),
        code(
            """
            subgroup_rows = [subgroup_effect("Overall", df)]
            subgroup_rows.append(subgroup_effect("New users", df[df["user_type"] == "new"]))
            subgroup_rows.append(subgroup_effect("Returning users", df[df["user_type"] == "returning"]))
            subgroup_rows.append(subgroup_effect("Mobile", df[df["device"] == "mobile"]))
            subgroup_rows.append(subgroup_effect("Desktop", df[df["device"] == "desktop"]))
            subgroup_rows.append(subgroup_effect("Tablet", df[df["device"] == "tablet"]))

            subgroup_effects = pd.DataFrame(subgroup_rows)
            interaction_model = smf.glm(
                "converted ~ C(group) * C(user_type) + C(device)",
                data=df,
                family=sm.families.Binomial(),
            ).fit()
            interaction_p = interaction_model.pvalues["C(group)[T.treatment]:C(user_type)[T.returning]"]

            display(style_table(subgroup_effects, {"effect_pp": "{:.2f}", "ci_low_pp": "{:.2f}", "ci_high_pp": "{:.2f}", "p_value": "{:.4f}", "control_rate": "{:.2f}", "treatment_rate": "{:.2f}", "users": "{:,.0f}"}))
            display(Markdown(f"Treatment-by-user-type interaction p-value = **{interaction_p:.4f}**. I do not have evidence that the effect truly differs between new and returning users."))
            """
        ),
        code(
            """
            sample_size_table = (
                df.groupby("user_type")["converted"]
                .mean()
                .reset_index(name="baseline_rate")
                .assign(
                    baseline_rate_pct=lambda frame: frame["baseline_rate"] * 100,
                    required_per_arm=lambda frame: frame["baseline_rate"].apply(sample_size_for_one_point_lift),
                    observed_users=lambda frame: frame["user_type"].map(df["user_type"].value_counts()),
                )
            )

            display(
                style_table(
                    sample_size_table[["user_type", "baseline_rate_pct", "required_per_arm", "observed_users"]],
                    {"baseline_rate_pct": "{:.2f}", "required_per_arm": "{:,.0f}", "observed_users": "{:,.0f}"},
                )
            )
            """
        ),
        code(
            """
            ordered_segments = subgroup_effects.iloc[::-1]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ordered_segments["effect_pp"],
                    y=ordered_segments["segment"],
                    mode="markers",
                    marker={"color": TREATMENT_COLOR, "size": 10},
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": ordered_segments["ci_high_pp"] - ordered_segments["effect_pp"],
                        "arrayminus": ordered_segments["effect_pp"] - ordered_segments["ci_low_pp"],
                    },
                    name="Effect estimate",
                )
            )
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(title="Subgroup Effects: Several Slices Are Too Noisy for Strong Claims", xaxis_title="Treatment lift (pp)", yaxis_title="")
            fig.update_yaxes(showgrid=False)
            fig.show()
            """
        ),
        md(
            """
            The right question is not "which subgroup has the biggest point estimate?" It is "which subgroup estimate is precise enough to support a product decision?" Here, several intervals still cross zero, and the user-type interaction is not significant. That is not a stable personalization story.
            """
        ),
        md(
            """
            ## Part 4. The Real Answer

            At this point the topline is no longer the story. The experiment is compromised, the durable effect is tiny, and the strongest-looking evidence is exactly the kind of evidence I should distrust.
            """
        ),
        code(
            """
            week4_row = weekly_effects.loc[weekly_effects["week"] == "Week 4"].iloc[0]
            desktop_anomaly = segment_summary.loc[segment_summary["device"].eq("desktop") & segment_summary["user_type"].eq("returning"), "treatment_share"].iloc[0]

            summary_text = f'''
            ### Stakeholder summary
            **Bottom line:** I would **not ship** this checkout redesign based on the current experiment. The topline lift is inflated by a randomization bug, and the device-adjusted effect is much smaller and less stable than the topline suggests.

            **What went wrong:** The experiment shows a significant SRM (p = **{srm_p:.5f}**). Returning desktop users landed in treatment **{desktop_anomaly:.2%}** of the time, which pulled more high-converting traffic into the treatment arm and created the Simpson's paradox effect. Once I standardize for device mix, the overall lift shrinks from **{overall_diff * 100:.2f} pp** to **{adjusted_diff * 100:.2f} pp**.

            **Why I do not trust the early win:** The device-adjusted lift starts at **{weekly_effects.loc[weekly_effects['week'] == 'Week 1', 'adjusted_effect_pp'].iloc[0]:.2f} pp** in week 1, turns negative in the middle of the run, and only ends at **{week4_row['adjusted_effect_pp']:.2f} pp** in week 4. That pattern is much more consistent with novelty and instability than with a clean product improvement.

            **Practical lens:** Even the naive topline lift (**{overall_diff * 100:.2f} pp**) is modest relative to the size of effect this test was powered to detect reliably (**~{mde_pp:.2f} pp**). This was not an obviously strong product win even before the audit concerns.

            **Recommendation:** Fix the assignment bug, rerun the test for at least four weeks, and evaluate the redesign only after the effect trajectory has settled rather than relying on the early spike.
            '''

            display(Markdown(summary_text))
            """
        ),
        md(
            """
            ### Put the key evidence on one page

            For stakeholders, I want one summary figure that shows the entire argument at a glance: the effect decay, the device-composition trap, and the noisy subgroup picture.
            """
        ),
        code(
            """
            forest_data = subgroup_effects.iloc[::-1]
            device_panel = conversion_by_device.assign(conversion_rate=lambda frame: frame["conversion_rate"] * 100)

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

            fig.add_trace(go.Scatter(x=weekly_effects["week"], y=weekly_effects["adjusted_effect_pp"], mode="lines+markers", line={"color": TREATMENT_COLOR, "width": 3}, name="Weekly effect"), row=1, col=1)
            fig.add_trace(go.Bar(x=device_panel[device_panel["group"] == "control"]["device"], y=device_panel[device_panel["group"] == "control"]["conversion_rate"], marker_color=CONTROL_COLOR, name="Control"), row=2, col=1)
            fig.add_trace(go.Bar(x=device_panel[device_panel["group"] == "treatment"]["device"], y=device_panel[device_panel["group"] == "treatment"]["conversion_rate"], marker_color=TREATMENT_COLOR, name="Treatment"), row=2, col=1)
            fig.add_trace(go.Scatter(x=forest_data["effect_pp"], y=forest_data["segment"], mode="markers", marker={"color": TREATMENT_COLOR, "size": 9}, error_x={"type": "data", "symmetric": False, "array": forest_data["ci_high_pp"] - forest_data["effect_pp"], "arrayminus": forest_data["effect_pp"] - forest_data["ci_low_pp"]}, name="Subgroup lift"), row=3, col=1)

            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", row=3, col=1)
            fig.update_layout(title="Experiment Summary: Why the Topline Lied", height=1100, barmode="group")
            fig.update_xaxes(title_text="Enrollment week", row=1, col=1)
            fig.update_yaxes(title_text="Lift (pp)", row=1, col=1, showgrid=False)
            fig.update_xaxes(title_text="Device", row=2, col=1)
            fig.update_yaxes(title_text="Conversion rate (%)", row=2, col=1, showgrid=False)
            fig.update_xaxes(title_text="Lift (pp)", row=3, col=1)
            fig.update_yaxes(title_text="", row=3, col=1, showgrid=False)
            fig.show()
            """
        ),
        md(
            """
            ## Part 5. Ground Truth Reveal

            This was simulated data with known failure modes baked in. I only reveal that design here, after the analysis, to show that the audit recovered the real story for the right reasons.
            """
        ),
        code(
            """
            control_truth = pd.DataFrame(
                ground_truth["control_conversion_rates"].items(),
                columns=["device", "control_rate"],
            )

            treatment_truth = (
                pd.DataFrame(ground_truth["treatment_conversion_rates"])
                .rename_axis("device")
                .reset_index()
            )

            display(style_table(control_truth, {"control_rate": "{:.2%}"}))
            display(style_table(treatment_truth, {"week_1": "{:.2%}", "week_2": "{:.2%}", "week_3": "{:.2%}", "week_4": "{:.2%}"}))

            reveal = ground_truth["assignment_mechanism"]
            display(
                Markdown(
                    f"The simulator biased **{reveal['segment']['user_type']} {reveal['segment']['device']}** users into treatment at **{reveal['treatment_probability']:.0%}** instead of 50%, while all other segments stayed at 50/50. "
                    f"I also delayed that segment's enrollment by **{ground_truth['biased_segment_enrollment_shift_days']} days** on average and added a short treatment pulse early in the run. "
                    "That combination is what creates the SRM, the misleading topline, and the unstable peeking story the audit recovered."
                )
            )
            """
        ),
    ]

    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }
    return notebook


def main() -> None:
    """Write the analysis notebook to notebooks/analysis.ipynb."""

    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "notebooks" / "analysis.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    with notebook_path.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)


if __name__ == "__main__":
    main()
