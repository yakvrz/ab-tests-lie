"""Simulate a flawed CartCo checkout experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_USERS = 50_000
EXPERIMENT_DAYS = 28
SIGNUP_LOOKBACK_DAYS = 730
CONTROL_COLOR = "#636EFA"
TREATMENT_COLOR = "#EF553B"


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the CartCo experiment simulation."""

    seed: int = SEED
    n_users: int = N_USERS
    experiment_days: int = EXPERIMENT_DAYS
    signup_lookback_days: int = SIGNUP_LOOKBACK_DAYS
    control_rates: dict[str, float] | None = None
    treatment_rates: dict[str, dict[str, float]] | None = None
    biased_segment_assignment: dict[str, object] | None = None
    biased_segment_enrollment_shift_days: int = 0
    device_probabilities: dict[str, float] | None = None
    user_type_given_device: dict[str, float] | None = None
    revenue_lognormal: dict[str, float] | None = None
    base_seed: int = 796136
    revenue_seed: int = 5
    treatment_week_adjustments: dict[str, float] | None = None
    early_novelty_pulse: dict[str, float] | None = None
    late_rebound_pulse: dict[str, float] | None = None
    treatment_user_type_offsets: dict[str, float] | None = None


def build_config() -> SimulationConfig:
    """Return the fixed ground-truth configuration used for the simulation."""

    return SimulationConfig(
        control_rates={
            "mobile": 0.08,
            "desktop": 0.18,
            "tablet": 0.12,
        },
        treatment_rates={
            "week_1": {"mobile": 0.095, "desktop": 0.19, "tablet": 0.135},
            "week_2": {"mobile": 0.088, "desktop": 0.183, "tablet": 0.125},
            "week_3": {"mobile": 0.083, "desktop": 0.178, "tablet": 0.121},
            "week_4": {"mobile": 0.081, "desktop": 0.175, "tablet": 0.119},
        },
        biased_segment_assignment={
            "segment": {"device": "desktop", "user_type": "returning"},
            "treatment_probability": 0.55,
            "all_other_segments_probability": 0.50,
        },
        biased_segment_enrollment_shift_days=8,
        device_probabilities={"mobile": 0.55, "desktop": 0.35, "tablet": 0.10},
        user_type_given_device={
            "mobile": 0.30,
            "desktop": 0.55,
            "tablet": 0.42,
        },
        revenue_lognormal={"mean": 3.5, "sigma": 0.8},
        treatment_week_adjustments={
            "week_1": 0.00172498,
            "week_2": -0.00018881,
            "week_3": -0.00138775,
            "week_4": -0.00077710,
        },
        early_novelty_pulse={"end_day": 4, "lift": 0.009},
        late_rebound_pulse={"start_day": 23, "lift": 0.0035},
        treatment_user_type_offsets={"new": -0.00029273, "returning": -0.005},
    )


def simulate_users(config: SimulationConfig) -> pd.DataFrame:
    """Generate the user-level experiment table."""

    rng = np.random.default_rng(config.base_seed)
    signup_rng = np.random.default_rng(config.base_seed + 1)

    user_id = np.arange(1, config.n_users + 1)
    device = rng.choice(
        ["mobile", "desktop", "tablet"],
        size=config.n_users,
        p=[config.device_probabilities["mobile"], config.device_probabilities["desktop"], config.device_probabilities["tablet"]],
    )
    user_type = sample_user_type(device, rng, config.user_type_given_device)
    day_enrolled = np.clip(rng.exponential(scale=8, size=config.n_users).astype(int) + 1, 1, config.experiment_days)
    late_segment = (device == "desktop") & (user_type == "returning")
    day_enrolled = np.clip(
        day_enrolled + np.where(late_segment, config.biased_segment_enrollment_shift_days, 0),
        1,
        config.experiment_days,
    )

    experiment_end = pd.Timestamp("2025-03-31")
    signup_offsets = signup_rng.integers(0, config.signup_lookback_days, size=config.n_users)
    signup_date = experiment_end - pd.to_timedelta(signup_offsets, unit="D")

    assignment_probability = np.full(config.n_users, 0.50)
    biased_segment = (device == "desktop") & (user_type == "returning")
    assignment_probability[biased_segment] = config.biased_segment_assignment["treatment_probability"]
    group = np.where(rng.random(config.n_users) < assignment_probability, "treatment", "control")

    conversion_probability = build_conversion_probability(
        group=group,
        device=device,
        user_type=user_type,
        day_enrolled=day_enrolled,
        control_rates=config.control_rates,
        treatment_rates=config.treatment_rates,
        treatment_week_adjustments=config.treatment_week_adjustments,
        early_novelty_pulse=config.early_novelty_pulse,
        late_rebound_pulse=config.late_rebound_pulse,
        treatment_user_type_offsets=config.treatment_user_type_offsets,
    )
    converted = (rng.random(config.n_users) < conversion_probability).astype(int)

    revenue = np.zeros(config.n_users, dtype=float)
    n_converted = converted.sum()
    revenue_rng = np.random.default_rng(config.revenue_seed)
    revenue[converted == 1] = revenue_rng.lognormal(
        mean=config.revenue_lognormal["mean"],
        sigma=config.revenue_lognormal["sigma"],
        size=n_converted,
    )

    users = pd.DataFrame(
        {
            "user_id": user_id,
            "group": group,
            "device": device,
            "user_type": user_type,
            "signup_date": signup_date,
            "day_enrolled": day_enrolled,
            "converted": converted,
            "revenue": revenue.round(2),
        }
    )

    return users.sort_values("user_id").reset_index(drop=True)


def sample_user_type(device: np.ndarray, rng: np.random.Generator, probs: dict[str, float]) -> np.ndarray:
    """Sample user type with device-specific returning rates."""

    returning_probability = np.select(
        [device == "mobile", device == "desktop", device == "tablet"],
        [probs["mobile"], probs["desktop"], probs["tablet"]],
    )
    return np.where(rng.random(device.size) < returning_probability, "returning", "new")


def build_conversion_probability(
    group: np.ndarray,
    device: np.ndarray,
    user_type: np.ndarray,
    day_enrolled: np.ndarray,
    control_rates: dict[str, float],
    treatment_rates: dict[str, dict[str, float]],
    treatment_week_adjustments: dict[str, float],
    early_novelty_pulse: dict[str, float],
    late_rebound_pulse: dict[str, float],
    treatment_user_type_offsets: dict[str, float],
) -> np.ndarray:
    """Return the conversion probability for each user."""

    week = week_label(day_enrolled)
    probability = np.zeros(group.size, dtype=float)

    for device_name, rate in control_rates.items():
        mask = (group == "control") & (device == device_name)
        probability[mask] = rate

    for week_name, rates in treatment_rates.items():
        for device_name, rate in rates.items():
            mask = (group == "treatment") & (week == week_name) & (device == device_name)
            probability[mask] = rate + treatment_week_adjustments[week_name]

    treatment_mask = group == "treatment"
    probability += np.where(
        treatment_mask & (user_type == "new"),
        treatment_user_type_offsets["new"],
        0.0,
    )
    probability += np.where(
        treatment_mask & (user_type == "returning"),
        treatment_user_type_offsets["returning"],
        0.0,
    )
    probability += np.where(
        treatment_mask & (day_enrolled <= early_novelty_pulse["end_day"]),
        early_novelty_pulse["lift"],
        0.0,
    )
    probability += np.where(
        treatment_mask & (day_enrolled >= late_rebound_pulse["start_day"]),
        late_rebound_pulse["lift"],
        0.0,
    )

    return np.clip(probability, 0.001, 0.999)


def week_label(day_enrolled: np.ndarray | pd.Series) -> np.ndarray:
    """Map experiment day to enrollment week labels."""

    bins = np.array([7, 14, 21, 28])
    week_index = np.digitize(day_enrolled, bins, right=True) + 1
    return np.array([f"week_{idx}" for idx in week_index])


def build_ground_truth(config: SimulationConfig) -> dict[str, object]:
    """Serialize ground-truth parameters for the notebook reveal."""

    return {
        "seed": config.seed,
        "n_users": config.n_users,
        "experiment_days": config.experiment_days,
        "signup_lookback_days": config.signup_lookback_days,
        "device_probabilities": config.device_probabilities,
        "user_type_given_device_returning_probability": config.user_type_given_device,
        "control_conversion_rates": config.control_rates,
        "treatment_conversion_rates": config.treatment_rates,
        "assignment_mechanism": config.biased_segment_assignment,
        "biased_segment_enrollment_shift_days": config.biased_segment_enrollment_shift_days,
        "revenue_distribution": {"distribution": "lognormal", **config.revenue_lognormal},
        "revenue_seed": config.revenue_seed,
        "treatment_week_adjustments": config.treatment_week_adjustments,
        "early_novelty_pulse": config.early_novelty_pulse,
        "late_rebound_pulse": config.late_rebound_pulse,
        "treatment_user_type_offsets": config.treatment_user_type_offsets,
        "visual_style": {"control": CONTROL_COLOR, "treatment": TREATMENT_COLOR},
    }


def print_diagnostics(users: pd.DataFrame) -> None:
    """Print headline metrics to help validate the simulated shape."""

    group_split = users["group"].value_counts(normalize=True).sort_index().mul(100).round(2)
    device_split = (
        users.groupby("group")["device"]
        .value_counts(normalize=True)
        .rename("share")
        .mul(100)
        .round(2)
        .reset_index()
        .pivot(index="group", columns="device", values="share")
        .reindex(["control", "treatment"])
    )
    conversion = users.groupby("group")["converted"].mean().mul(100).round(2)
    revenue = users.groupby("group")["revenue"].mean().round(2)
    subgroup_conversion = (
        users.groupby(["user_type", "group"])["converted"]
        .mean()
        .mul(100)
        .round(2)
        .unstack()
    )

    print("Observed group split (%):")
    print(group_split.to_string())
    print("\nObserved device share by group (%):")
    print(device_split.to_string())
    print("\nObserved conversion rate (%):")
    print(conversion.to_string())
    print("\nObserved revenue per user:")
    print(revenue.to_string())
    print("\nObserved conversion by user type (%):")
    print(subgroup_conversion.to_string())


def save_outputs(users: pd.DataFrame, config: SimulationConfig) -> None:
    """Persist the simulated data and ground-truth metadata."""

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    users.to_csv(data_dir / "ab_test_raw.csv", index=False)
    with (data_dir / "ground_truth.json").open("w", encoding="utf-8") as handle:
        json.dump(build_ground_truth(config), handle, indent=2)


def main() -> None:
    """Run the full simulation pipeline and write outputs to disk."""

    config = build_config()
    users = simulate_users(config)
    save_outputs(users, config)
    print_diagnostics(users)


if __name__ == "__main__":
    main()
