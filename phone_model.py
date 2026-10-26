"""Smartphone value-for-money recommendation model (Business Analytics DA2).

Used by both the Jupyter notebook and the Streamlit app, so the results in the
report and the live app always come from the same code.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEGMENT_BINS = [0, 15000, 35000, np.inf]
SEGMENT_LABELS = ["Budget (< ₹15k)", "Mid-range (₹15k–35k)", "Premium (> ₹35k)"]

# Specification columns used to predict the "fair price"
REG_FEATURES = [
    "ram_capacity", "internal_memory", "battery_gt", "fast_charging",
    "refresh_rate", "primary_camera_rear", "primary_camera_front",
    "screen_size", "pixels_mp", "processor_speed", "num_cores",
    "5G_or_not", "extended_memory_available", "is_ios",
]


def load_raw(path):
    return pd.read_csv(path)


def clean(raw, min_brand_count=20):
    """Preprocess the raw dataset. Returns (clean_df, log_of_steps)."""
    df = raw.copy()
    log = {}

    log["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # Missing values
    log["missing_before"] = df.isna().sum()[df.isna().sum() > 0].to_dict()
    df["rating_was_missing"] = df["avg_rating"].isna().astype(int)
    df["avg_rating"] = df.groupby("brand_name")["avg_rating"].transform(
        lambda s: s.fillna(s.median()))
    df["avg_rating"] = df["avg_rating"].fillna(df["avg_rating"].median())
    df["fast_charging"] = df["fast_charging"].fillna(0)  # no fast charging
    for col in ["processor_speed", "battery_capacity", "num_cores",
                "primary_camera_front", "primary_camera_rear"]:
        df[col] = df[col].fillna(df[col].median())
    for col in ["processor_brand", "os"]:
        df[col] = df[col].fillna("unknown")
    log["missing_after"] = int(df.isna().sum().sum())

    # Feature engineering
    df["battery_gt"] = df["battery_capacity"] / 1000          # battery in 1000 mAh
    df["pixels_mp"] = df["resolution_height"] * df["resolution_width"] / 1e6
    df["is_ios"] = (df["os"] == "ios").astype(float)           # platform premium control
    df["log_price"] = np.log(df["price"])
    counts = df["brand_name"].value_counts()
    big = counts[counts >= min_brand_count].index
    df["brand_group"] = np.where(df["brand_name"].isin(big), df["brand_name"], "other")
    df["segment"] = pd.cut(df["price"], SEGMENT_BINS, labels=SEGMENT_LABELS)

    # Outliers (IQR rule on price) are flagged, not removed
    q1, q3 = df["price"].quantile([0.25, 0.75])
    df["price_outlier"] = (df["price"] > q3 + 1.5 * (q3 - q1)).astype(int)
    log["price_outliers_flagged"] = int(df["price_outlier"].sum())
    return df.reset_index(drop=True), log


def fit_price_model(df, features=None, use_brand=False, robust=True):
    """OLS of log(price) on specifications. Returns the fitted statsmodels result.

    robust=True reports heteroscedasticity-robust (HC3) standard errors; the
    coefficients and fitted values are identical to plain OLS.
    """
    feats = features or REG_FEATURES
    X = df[feats].astype(float)
    if use_brand:
        X = pd.concat([X, pd.get_dummies(df["brand_group"], drop_first=True,
                                         dtype=float)], axis=1)
    res = sm.OLS(df["log_price"], sm.add_constant(X))
    return res.fit(cov_type="HC3") if robust else res.fit()


def add_value_score(df, model):
    """Fair price from specs; value score = (fair - actual) / fair."""
    out = df.copy()
    X = sm.add_constant(out[REG_FEATURES].astype(float), has_constant="add")
    out["fair_price"] = np.exp(model.predict(X))
    out["value_score"] = (out["fair_price"] - out["price"]) / out["fair_price"]
    return out


def add_clusters(df, k=3, seed=42):
    feats = ["log_price", "ram_capacity", "internal_memory", "battery_gt",
             "refresh_rate", "primary_camera_rear", "pixels_mp"]
    Z = StandardScaler().fit_transform(df[feats])
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    out = df.copy()
    labels = km.labels_
    # name clusters by average price (low -> high)
    order = out.groupby(labels)["price"].mean().sort_values().index.tolist()
    names = {c: n for c, n in zip(order, ["Budget", "Mid-range", "Premium"][:k])}
    out["cluster"] = [names[c] for c in labels]
    return out, Z


def _pct(s):
    return s.rank(pct=True)


def add_feature_scores(df):
    """0-1 scores (percentile rank across all phones) for the four priorities."""
    out = df.copy()
    out["score_performance"] = (_pct(out["ram_capacity"]) + _pct(out["processor_speed"])
                                + _pct(out["num_cores"]) + _pct(out["internal_memory"])) / 4
    out["score_camera"] = (2 * _pct(out["primary_camera_rear"])
                           + _pct(out["primary_camera_front"])) / 3
    out["score_battery"] = (2 * _pct(out["battery_capacity"])
                            + _pct(out["fast_charging"])) / 3
    out["score_display"] = (_pct(out["refresh_rate"]) + _pct(out["pixels_mp"])
                            + _pct(out["screen_size"])) / 3
    return out


def recommend(df, budget, w_perf=1, w_cam=1, w_batt=1, w_disp=1, w_value=1,
              need_5g=False, brands=None, min_rating=0.0, top_n=5):
    """Return the top_n phones within budget.

    final score = (weighted mean of the four feature scores) * (1 - v) + value_component * v
    where v = share of weight given to value for money.
    """
    pool = df[df["price"] <= budget].copy()
    if need_5g:
        pool = pool[pool["5G_or_not"] == 1]
    if brands:
        pool = pool[pool["brand_name"].isin(brands)]
    pool = pool[pool["avg_rating"] >= min_rating]
    if pool.empty:
        return pool
    w = np.array([w_perf, w_cam, w_batt, w_disp], dtype=float)
    total = w_value + w.sum()
    if total == 0:
        w, w_value, total = np.ones(4), 1.0, 5.0
    fit = (pool[["score_performance", "score_camera", "score_battery",
                 "score_display"]].values @ w) / max(w.sum(), 1e-9)
    value_component = _pct(pool["value_score"]).values
    v = w_value / total
    pool["match_score"] = 100 * ((1 - v) * fit + v * value_component)
    pool = pool.sort_values("match_score", ascending=False).head(top_n)
    pool["why"] = pool.apply(_reason, axis=1)
    return pool


def _reason(r):
    bits = []
    if r["value_score"] > 0.05:
        bits.append(f"priced {r['value_score']*100:.0f}% below its spec-based fair price")
    elif r["value_score"] < -0.05:
        bits.append(f"priced {abs(r['value_score'])*100:.0f}% above its spec-based fair price")
    else:
        bits.append("priced close to its fair price")
    strengths = {"performance": r["score_performance"], "camera": r["score_camera"],
                 "battery": r["score_battery"], "display": r["score_display"]}
    best = max(strengths, key=strengths.get)
    bits.append(f"strongest on {best}")
    return "; ".join(bits)
