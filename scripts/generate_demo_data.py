from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def main():
    rng = np.random.default_rng(42)
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_rows = 8000
    transaction_id = np.arange(100000, 100000 + n_rows)
    transaction_dt = np.arange(n_rows)

    transaction_amt = rng.lognormal(mean=4.0, sigma=0.9, size=n_rows)
    dist1 = rng.normal(loc=20, scale=8, size=n_rows)
    dist2 = rng.normal(loc=15, scale=5, size=n_rows)
    d1 = rng.normal(loc=5, scale=2, size=n_rows)
    d2 = rng.normal(loc=10, scale=3, size=n_rows)
    c1 = rng.poisson(lam=3, size=n_rows)
    c2 = rng.poisson(lam=5, size=n_rows)
    v258 = rng.normal(loc=0.0, scale=1.0, size=n_rows)
    v259 = rng.normal(loc=0.0, scale=1.0, size=n_rows)
    card1 = rng.integers(1000, 2000, size=n_rows)
    card2 = rng.integers(100, 600, size=n_rows)
    card3 = rng.integers(100, 200, size=n_rows)
    card5 = rng.integers(100, 300, size=n_rows)
    addr1 = rng.integers(100, 600, size=n_rows)
    addr2 = rng.integers(10, 100, size=n_rows)

    card4 = rng.choice(["visa", "mastercard", "discover", "american express"], size=n_rows)
    card6 = rng.choice(["credit", "debit", "charge"], size=n_rows, p=[0.45, 0.5, 0.05])
    product_cd = rng.choice(["W", "C", "R", "H", "S"], size=n_rows)
    p_emaildomain = rng.choice(
        [f"domain{i}.com" for i in range(120)] + ["gmail.com", "yahoo.com", "hotmail.com"],
        size=n_rows,
    )
    r_emaildomain = rng.choice(
        [f"merchant{i}.com" for i in range(150)] + ["bank.com", "service.com"],
        size=n_rows,
    )

    # Introduce time-based shift in later transactions to simulate emerging fraud patterns.
    late_period = transaction_dt > int(n_rows * 0.8)
    card4[late_period & (rng.random(n_rows) < 0.25)] = "emerging_card_network"
    product_cd[late_period & (rng.random(n_rows) < 0.2)] = "Z"

    raw_score = (
        -6.0
        + 0.0035 * transaction_amt
        + 0.7 * (card6 == "charge").astype(float)
        + 0.55 * (product_cd == "R").astype(float)
        + 0.45 * (p_emaildomain == "gmail.com").astype(float)
        + 0.7 * late_period.astype(float)
        + 1.6 * ((card4 == "emerging_card_network") & late_period).astype(float)
        + 0.15 * (c2 > 7).astype(float)
        + 0.1 * np.maximum(v258, 0)
        + 0.15 * np.maximum(v259, 0)
    )
    fraud_probability = sigmoid(raw_score)
    is_fraud = rng.binomial(1, np.clip(fraud_probability, 0.001, 0.95))

    transaction = pd.DataFrame(
        {
            "TransactionID": transaction_id,
            "TransactionDT": transaction_dt,
            "TransactionAmt": transaction_amt.round(2),
            "dist1": dist1.round(3),
            "dist2": dist2.round(3),
            "D1": d1.round(3),
            "D2": d2.round(3),
            "C1": c1,
            "C2": c2,
            "V258": v258.round(4),
            "V259": v259.round(4),
            "card1": card1,
            "card2": card2,
            "card3": card3,
            "card4": card4,
            "card5": card5,
            "card6": card6,
            "addr1": addr1,
            "addr2": addr2,
            "ProductCD": product_cd,
            "P_emaildomain": p_emaildomain,
            "R_emaildomain": r_emaildomain,
            "isFraud": is_fraud,
        }
    )

    for column in ["dist1", "dist2", "D1", "D2", "P_emaildomain", "R_emaildomain", "card6"]:
        mask = rng.random(n_rows) < 0.12
        transaction.loc[mask, column] = np.nan

    identity = pd.DataFrame(
        {
            "TransactionID": transaction_id,
            "DeviceType": rng.choice(["desktop", "mobile", "tablet"], size=n_rows),
            "DeviceInfo": rng.choice([f"device_{i}" for i in range(80)], size=n_rows),
            "id_12": rng.choice(["Found", "NotFound"], size=n_rows, p=[0.7, 0.3]),
            "id_15": rng.choice(["New", "Found", "Unknown"], size=n_rows),
            "id_30": rng.choice(["Windows", "iOS", "Android", "MacOS", "Linux"], size=n_rows),
            "id_31": rng.choice(["chrome", "safari", "firefox", "edge"], size=n_rows),
            "id_33": rng.choice(["1920x1080", "1366x768", "1280x720", "1536x864"], size=n_rows),
            "id_36": rng.choice(["T", "F"], size=n_rows),
        }
    )
    for column in ["DeviceType", "DeviceInfo", "id_30"]:
        mask = rng.random(n_rows) < 0.1
        identity.loc[mask, column] = np.nan

    transaction.to_csv(output_dir / "train_transaction.csv", index=False)
    identity.to_csv(output_dir / "train_identity.csv", index=False)

    summary = {
        "rows": n_rows,
        "fraud_rate": float(transaction["isFraud"].mean()),
        "late_period_fraud_rate": float(transaction.loc[late_period, "isFraud"].mean()),
        "early_period_fraud_rate": float(transaction.loc[~late_period, "isFraud"].mean()),
    }
    print(summary)


if __name__ == "__main__":
    main()
