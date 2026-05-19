from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import DATA_PATH, ensure_directories


RANDOM_SEED = 42


def generate_financial_risk_data(
    num_samples: int = 5000,
    random_seed: int = RANDOM_SEED,
    output_path: Path | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    ensure_directories()
    target_path = output_path or DATA_PATH

    age = rng.integers(21, 61, size=num_samples)
    employment_years = np.clip(age - 21 - rng.integers(0, 12, size=num_samples), 0, 40)
    credit_score = np.clip(rng.normal(660, 85, size=num_samples), 300, 850).round(0)
    income = np.clip(rng.lognormal(mean=11.6, sigma=0.45, size=num_samples), 50000, 800000)
    loan_term = rng.choice([12, 24, 36, 48, 60], size=num_samples, p=[0.12, 0.22, 0.28, 0.2, 0.18])
    has_house = rng.binomial(1, p=np.clip((income - 60000) / 400000 + 0.25, 0.15, 0.85))
    has_car = rng.binomial(1, p=np.clip((income - 50000) / 350000 + 0.35, 0.2, 0.9))

    base_loan = income * rng.uniform(0.15, 0.85, size=num_samples)
    asset_adjustment = (1 - 0.08 * has_house - 0.04 * has_car)
    loan_amount = np.clip(base_loan * asset_adjustment + rng.normal(30000, 25000, size=num_samples), 10000, 600000)

    debt_to_income_ratio = np.clip(
        loan_amount / np.maximum(income, 1) * rng.uniform(0.35, 0.95, size=num_samples),
        0.05,
        0.95,
    )

    overdue_lambda = np.clip(
        1.8
        - (credit_score - 300) / 350
        + debt_to_income_ratio * 1.5
        + (income < 120000).astype(float) * 0.4,
        0.05,
        4.5,
    )
    overdue_times = np.clip(rng.poisson(overdue_lambda), 0, 8)

    risk_score = (
        2.2 * (650 - credit_score) / 120
        + 2.1 * (debt_to_income_ratio - 0.35)
        + 0.35 * overdue_times
        + 0.9 * (loan_amount / np.maximum(income, 1) - 1.0)
        + 0.6 * (120000 / np.maximum(income, 50000) - 0.7)
        + 0.35 * (loan_term / 60)
        - 0.25 * employment_years / 10
        - 0.3 * has_house
        - 0.15 * has_car
        + 0.15 * (30 - age) / 10
        - 1.55
    )
    default_probability = 1.0 / (1.0 + np.exp(-risk_score))
    default = rng.binomial(1, default_probability)

    df = pd.DataFrame(
        {
            "age": age.astype(int),
            "income": np.round(income, 2),
            "loan_amount": np.round(loan_amount, 2),
            "credit_score": credit_score.astype(int),
            "employment_years": employment_years.astype(int),
            "debt_to_income_ratio": np.round(debt_to_income_ratio, 3),
            "overdue_times": overdue_times.astype(int),
            "loan_term": loan_term.astype(int),
            "has_house": has_house.astype(int),
            "has_car": has_car.astype(int),
            "default": default.astype(int),
        }
    )

    df.to_csv(target_path, index=False, encoding="utf-8")
    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    print("Financial risk dataset generated successfully.")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(df.head())


if __name__ == "__main__":
    dataset = generate_financial_risk_data()
    print_dataset_summary(dataset)
    print(f"Saved to: {DATA_PATH}")
