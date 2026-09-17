"""
01_generate_dataset.py
Synthetic Alternative Credit Scoring Dataset - Nigeria's Informal Economy
Author: Generated for Mariam Tajudeen Sulaiman's MIT Project (Chapter 4)

This script synthesizes a realistic dataset of informal-economy borrowers using
the alternative financial indicators identified in Chapter 2 (mobile money
transaction behaviour, savings patterns, repayment history, utility/rent
payment timeliness, and social-capital proxies), grounded in the Stiglitz &
Weiss (1981) information asymmetry framework: borrowers with weaker, noisier
alternative-data signals are harder for a lender to correctly price, which is
exactly the "adverse selection" gap alternative credit scoring is meant to close.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

N = 3000  # number of informal-economy borrower profiles

business_types = ["Trading", "Farming", "Transportation", "Craft/Artisan", "Freelance Services"]
business_type = rng.choice(business_types, size=N, p=[0.32, 0.22, 0.18, 0.16, 0.12])

# --- Core alternative financial indicators -------------------------------------------------

years_in_business = np.clip(rng.gamma(shape=2.2, scale=2.0, size=N), 0.2, 25)

# Mobile money / digital footprint
mm_txn_frequency = np.clip(rng.normal(18, 8, N), 1, 60)               # transactions/month
mm_txn_volume = np.clip(rng.lognormal(mean=10.2, sigma=0.55, size=N), 5000, 600000)  # naira/month
airtime_topup_consistency = np.clip(rng.beta(5, 2, N), 0, 1)          # 0-1 regularity score

# Savings behaviour
savings_frequency = np.clip(rng.normal(6, 3, N), 0, 20)               # deposits/month
avg_savings_balance = np.clip(rng.lognormal(mean=9.5, sigma=0.7, size=N), 1000, 400000)

# Bill / rent payment timeliness (proxy for repayment discipline)
utility_payment_timeliness = np.clip(rng.beta(6, 2.5, N), 0, 1)
rent_payment_timeliness = np.clip(rng.beta(5.5, 2.5, N), 0, 1)

# Prior informal credit / cooperative repayment history
repayment_history_score = np.clip(rng.beta(5, 2.2, N), 0, 1)
avg_payment_delay_days = np.clip(rng.gamma(shape=2.0, scale=3.0, size=N), 0, 60)

# Income & social capital proxies
income_consistency_score = np.clip(rng.beta(4.5, 2.5, N), 0, 1)
social_capital_score = np.clip(rng.beta(4.0, 2.5, N), 0, 1)           # peer/network trust proxy
digital_wallet_balance_volatility = np.clip(rng.gamma(shape=2.0, scale=0.18, size=N), 0.01, 2.0)

df = pd.DataFrame({
    "business_type": business_type,
    "years_in_business": years_in_business,
    "mobile_money_txn_frequency": mm_txn_frequency,
    "mobile_money_txn_volume_naira": mm_txn_volume,
    "airtime_topup_consistency": airtime_topup_consistency,
    "savings_frequency": savings_frequency,
    "avg_savings_balance_naira": avg_savings_balance,
    "utility_payment_timeliness": utility_payment_timeliness,
    "rent_payment_timeliness": rent_payment_timeliness,
    "repayment_history_score": repayment_history_score,
    "avg_payment_delay_days": avg_payment_delay_days,
    "income_consistency_score": income_consistency_score,
    "social_capital_score": social_capital_score,
    "wallet_balance_volatility": digital_wallet_balance_volatility,
})

# --- Latent creditworthiness / default-generating process ---------------------------------
# Weighted combination of standardized alternative indicators + noise (information asymmetry)
def z(x):
    return (x - x.mean()) / x.std()

latent_score = (
    -1.35 * z(df["repayment_history_score"])
    -0.95 * z(df["utility_payment_timeliness"])
    -0.85 * z(df["rent_payment_timeliness"])
    -0.70 * z(df["income_consistency_score"])
    -0.55 * z(df["airtime_topup_consistency"])
    -0.50 * z(df["savings_frequency"])
    -0.45 * z(np.log1p(df["avg_savings_balance_naira"]))
    -0.40 * z(np.log1p(df["mobile_money_txn_volume_naira"]))
    -0.35 * z(df["social_capital_score"])
    -0.30 * z(df["years_in_business"])
    +0.60 * z(df["avg_payment_delay_days"])
    +0.30 * z(df["wallet_balance_volatility"])
)

# business-type structural risk effect (captures sectoral cash-flow volatility)
sector_risk = df["business_type"].map({
    "Trading": -0.05, "Farming": 0.25, "Transportation": 0.05,
    "Craft/Artisan": -0.10, "Freelance Services": -0.15
}).astype(float)

# --- Non-linear / interaction effects -------------------------------------------------
# Real informal-sector risk is rarely a clean linear combination: thresholds and
# interactions between indicators matter (this is precisely why ensemble/tree methods
# are expected, per the literature review, to outperform linear models).
z_delay = z(df["avg_payment_delay_days"])
z_savings_freq = z(df["savings_frequency"])
z_years = z(df["years_in_business"])
z_income = z(df["income_consistency_score"])
z_vol = z(df["wallet_balance_volatility"])

# Threshold / non-monotonic effect: very new businesses (<1 yr) combined with high
# payment delay are disproportionately risky (compounding effect), not just additive.
compounding_risk = 2.10 * (z_delay * (df["years_in_business"] < 1.5).astype(float))

# Interaction: low savings discipline AND high wallet volatility together signal
# cash-flow fragility far beyond either indicator alone.
fragility_interaction = 1.80 * (np.maximum(0, -z_savings_freq) * np.maximum(0, z_vol))

# Non-monotonic effect of repayment history: only very low deciles are strongly
# predictive (a threshold/step effect rather than a smooth linear one).
repay_threshold_effect = -1.45 * (df["repayment_history_score"] < 0.35).astype(float)

# Interaction: inconsistent income combined with short business tenure compounds risk
tenure_income_interaction = 1.40 * (np.maximum(0, -z_income) * np.maximum(0, -z_years))

nonlinear_term = compounding_risk + fragility_interaction + repay_threshold_effect + tenure_income_interaction

noise = rng.normal(0, 1.0, N)   # irreducible information-asymmetry noise
default_logit = -0.95 + latent_score + sector_risk + nonlinear_term + noise
default_prob = 1 / (1 + np.exp(-default_logit))
default = rng.binomial(1, default_prob)

df["default"] = default

print("Dataset shape:", df.shape)
print("\nDefault rate:", round(df["default"].mean(), 4))
print("\nClass balance:\n", df["default"].value_counts())
print("\n", df.describe().T)

df.to_csv("credit_scoring_dataset.csv", index=False)
print("\nSaved credit_scoring_dataset.csv")
