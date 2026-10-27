"""Reproducible SYNTHETIC training data (NOT real user transactions).

Scenario: a month has a budget, some money already spent, some days left.
Projected end-of-month spend = spent + days_remaining * blended daily rate, where
the blended rate is 75% recent daily average + 25% last month's average daily rate,
plus a little random noise. Label 1 = projected spend exceeds the budget.
"""
import numpy as np

FEATURES = ["monthly_budget", "spent_so_far", "days_remaining",
            "avg_daily_spending", "previous_period_spending"]


def generate_dataset(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    budget = rng.uniform(500, 5000, n)
    days_left = rng.integers(1, 31, n).astype(float)
    days_gone = 30 - days_left
    avg_daily = budget / 30 * rng.uniform(0.3, 1.8, n)
    spent = avg_daily * days_gone * rng.uniform(0.8, 1.2, n)
    previous = budget * rng.uniform(0.5, 1.5, n)
    rate = 0.75 * avg_daily + 0.25 * previous / 30
    projected = spent + days_left * rate + rng.normal(0, 0.03, n) * budget
    y = (projected > budget).astype(float)
    X = np.column_stack([budget, spent, days_left, avg_daily, previous])
    return X, y
