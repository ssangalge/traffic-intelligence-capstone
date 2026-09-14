"""
unsupervised_learning.py

Unsupervised learning on the traffic dataset:
  - K-means clustering: groups hours into natural traffic/weather "regimes"
    based on time and weather features, without using the proxy risk label.
  - Association rule mining (Apriori): finds frequent co-occurring patterns
    among categorical conditions (e.g. weather, time-of-day bucket, risk level).

Usage:
    python unsupervised_learning.py

Input:  risk_labeled_data.csv (from proxy_risk_label.py)
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from logging_setup import configure_logging

logger = logging.getLogger(__name__)

INPUT_CSV_PATH = Path(__file__).parent / "risk_labeled_data.csv"
RANDOM_STATE = 42
N_CLUSTERS = 4


def load_data(csv_path: Path) -> pd.DataFrame:
    logger.info(f"Loading risk-labeled data from {csv_path}")
    if not csv_path.exists():
        logger.error(f"Data file not found at {csv_path}. Run proxy_risk_label.py first.")
        raise FileNotFoundError(f"Expected data at {csv_path}, but it does not exist.")

    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    logger.info(f"Loaded {len(df):,} rows.")
    return df


def run_kmeans_clustering(df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    """Cluster rows using hour, temperature, and traffic_volume."""
    logger.info(f"--- K-means clustering (k={n_clusters}) ---")

    cluster_features = df[["hour", "temp_celsius", "traffic_volume"]].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled_features)

    logger.info(f"K-means fit complete. Cluster sizes: {df['cluster'].value_counts().sort_index().to_dict()}")

    logger.info("Cluster profiles (mean values per cluster):")
    profile = df.groupby("cluster")[["hour", "temp_celsius", "traffic_volume"]].mean().round(1)
    for cluster_id, row in profile.iterrows():
        logger.info(
            f"  Cluster {cluster_id}: avg hour={row['hour']}, "
            f"avg temp={row['temp_celsius']}C, avg volume={row['traffic_volume']}"
        )

    risk_by_cluster = df.groupby("cluster")["proxy_risk_label"].apply(
        lambda x: (x == "High").mean()
    )
    logger.info(f"High-risk rate by cluster: {risk_by_cluster.round(3).to_dict()}")

    return df


def run_association_rules(df: pd.DataFrame) -> None:
    """Mine association rules among categorical conditions using Apriori."""
    logger.info("--- Association rule mining (Apriori) ---")

    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        logger.error("mlxtend is required for association rule mining. Install with: pip install mlxtend")
        raise

    def hour_bucket(hour: int) -> str:
        if 6 <= hour < 10:
            return "morning_rush"
        elif 10 <= hour < 16:
            return "midday"
        elif 16 <= hour < 19:
            return "evening_rush"
        else:
            return "overnight"

    transactions = []
    for _, row in df.iterrows():
        items = [
            f"hour_bucket={hour_bucket(row['hour'])}",
            f"weather={row['weather_main']}",
            f"risk={row['proxy_risk_label']}",
            f"weekend={row['is_weekend']}",
        ]
        transactions.append(items)

    logger.info(f"Built {len(transactions):,} transactions for association rule mining.")

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = apriori(transaction_df, min_support=0.02, use_colnames=True)
    logger.info(f"Found {len(frequent_itemsets)} frequent itemsets (min_support=0.02).")

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    rules = rules.sort_values("lift", ascending=False)

    logger.info(f"Generated {len(rules)} association rules (min_confidence=0.5).")
    logger.info("Top 5 rules by lift:")
    for _, rule in rules.head(5).iterrows():
        antecedent = ", ".join(list(rule["antecedents"]))
        consequent = ", ".join(list(rule["consequents"]))
        logger.info(
            f"  IF ({antecedent}) THEN ({consequent}) "
            f"[support={rule['support']:.3f}, confidence={rule['confidence']:.3f}, lift={rule['lift']:.2f}]"
        )


def run_unsupervised_learning(input_csv_path: Path = INPUT_CSV_PATH) -> None:
    logger.info("=== Starting unsupervised learning ===")

    df = load_data(input_csv_path)
    df = run_kmeans_clustering(df)
    run_association_rules(df)

    logger.info("=== Unsupervised learning completed successfully ===")


if __name__ == "__main__":
    configure_logging()
    try:
        run_unsupervised_learning()
    except Exception as e:
        logger.error(f"Unsupervised learning failed: {e}")
        sys.exit(1)