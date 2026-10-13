from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


NUMERIC_FEATURES = [
    "amount",
    "transaction_hour",
    "sender_previous_balance",
    "sender_new_balance",
    "receiver_previous_balance",
    "receiver_new_balance",
    "transactions_last_hour",
    "account_age_days",
]
CATEGORICAL_FEATURES = ["transaction_type", "device_type", "location"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_fraud"


def generate_synthetic_transactions(rows=3200, seed=42):
    """Create a reproducible, educational synthetic payment dataset."""
    rng = np.random.default_rng(seed)
    transaction_type = rng.choice(
        ["TRANSFER", "PAYMENT", "CASH_OUT", "CASH_IN", "DEBIT"],
        rows,
        p=[0.25, 0.32, 0.18, 0.12, 0.13],
    )
    amount = np.clip(rng.lognormal(5.35, 1.1, rows), 1, 180000).round(2)
    hour = rng.integers(0, 24, rows)
    sender_previous = np.clip(rng.lognormal(8.5, 1.15, rows), 0, 500000).round(2)
    sender_new = np.maximum(0, sender_previous - amount + rng.normal(0, 18, rows)).round(2)
    receiver_previous = np.clip(rng.lognormal(8.0, 1.2, rows), 0, 500000).round(2)
    receiver_new = (receiver_previous + amount + rng.normal(0, 18, rows)).clip(0).round(2)
    transactions_last_hour = rng.poisson(1.6, rows).clip(0, 16)
    account_age = np.maximum(1, rng.gamma(3.0, 290, rows)).clip(1, 5000).astype(int)
    device = rng.choice(["Mobile", "Desktop", "Tablet", "Unknown"], rows, p=[0.68, 0.20, 0.09, 0.03])
    location = rng.choice(
        ["Local", "Domestic", "International", "VPN / Proxy"],
        rows,
        p=[0.56, 0.30, 0.10, 0.04],
    )

    log_odds = np.full(rows, -3.25)
    log_odds += np.isin(transaction_type, ["TRANSFER", "CASH_OUT"]) * 1.35
    log_odds += (amount > 45000) * 1.55 + (amount > 100000) * 0.75
    log_odds += np.isin(hour, [0, 1, 2, 3, 4]) * 1.0
    log_odds += (transactions_last_hour >= 5) * 1.45
    log_odds += (transactions_last_hour >= 9) * 1.0
    log_odds += (account_age < 45) * 1.25
    log_odds += np.isin(location, ["International", "VPN / Proxy"]) * 1.5
    log_odds += (device == "Unknown") * 0.8
    log_odds += ((sender_previous - sender_new) > sender_previous * 0.96) * 0.65
    log_odds += rng.normal(0, 0.5, rows)
    probability = 1 / (1 + np.exp(-log_odds))
    is_fraud = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "amount": amount,
            "transaction_type": transaction_type,
            "transaction_hour": hour,
            "sender_previous_balance": sender_previous,
            "sender_new_balance": sender_new,
            "receiver_previous_balance": receiver_previous,
            "receiver_new_balance": receiver_new,
            "transactions_last_hour": transactions_last_hour,
            "device_type": device,
            "location": location,
            "account_age_days": account_age,
            TARGET: is_fraud,
        }
    )


def ensure_sample_dataset(dataset_path, regenerate=False):
    path = Path(dataset_path)
    if regenerate or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_transactions().to_csv(path, index=False)
    return pd.read_csv(path).drop_duplicates().dropna(subset=FEATURES + [TARGET])


def create_eda_artifacts(data, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "duplicate_rows_removed": 0,
        "missing_values_after_cleaning": int(data.isna().sum().sum()),
        "fraud_rate": round(float(data[TARGET].mean()), 4),
        "class_counts": {str(label): int(count) for label, count in data[TARGET].value_counts().sort_index().items()},
        "amount": {
            "mean": round(float(data["amount"].mean()), 2),
            "median": round(float(data["amount"].median()), 2),
            "maximum": round(float(data["amount"].max()), 2),
        },
        "transaction_types": {
            str(label): int(count)
            for label, count in data["transaction_type"].value_counts().sort_index().items()
        },
    }
    (output_path / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sns.set_theme(style="whitegrid", palette=["#5b8f75", "#dc6858"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    sns.countplot(data=data, x=TARGET, hue=TARGET, legend=False, ax=axes[0])
    axes[0].set(title="Synthetic label balance", xlabel="Fraud label (0 = legitimate)", ylabel="Transactions")
    sns.boxplot(data=data, x=TARGET, y="amount", hue=TARGET, legend=False, showfliers=False, ax=axes[1])
    axes[1].set(title="Amount by synthetic class", xlabel="Fraud label", ylabel="Transaction amount (USD)")
    axes[1].set_yscale("log")
    figure.savefig(output_path / "eda_class_and_amount.png", dpi=140, facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    type_rates = data.groupby("transaction_type", observed=True)[TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=type_rates.index, y=type_rates.values, color="#4c7862", ax=axis)
    axis.set(title="Synthetic fraud-label rate by transaction type", xlabel="Transaction type", ylabel="Label rate")
    axis.set_ylim(0, max(0.1, float(type_rates.max()) * 1.15))
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    figure.savefig(output_path / "eda_type_rates.png", dpi=140, facecolor="white")
    plt.close(figure)