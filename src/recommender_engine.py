import io
import json
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

#Core mining & algorithm 
EVENT_WEIGHTS = {"view": 1.0, "wishlist": 2.0, "cart": 3.0, "purchase": 5.0}

def _quantile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    ranked = series.rank(method="first", pct=True)
    score = np.ceil(ranked * 5).astype(int).clip(1, 5)
    return score if higher_is_better else 6 - score

def _standardise(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    std = float(values.std(ddof=0))
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std

def build_rfm_table(customers: pd.DataFrame, events: pd.DataFrame, as_of_date: pd.Timestamp | None = None) -> pd.DataFrame:
    purchases = events.loc[events["event_type"].eq("purchase")].copy()
    if purchases.empty:
        raise ValueError("No purchase events found for RFM computation.")

    if as_of_date is None:
        as_of_date = purchases["timestamp"].max().normalize() + pd.Timedelta(days=1)
    else:
        as_of_date = pd.Timestamp(as_of_date)

    purchase_summary = (
        purchases.groupby("customer_id")
        .agg(
            last_purchase=("timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("revenue", "sum"),
            purchase_events=("event_id", "nunique"),
        )
        .reset_index()
    )
    purchase_summary["recency"] = (as_of_date.normalize() - purchase_summary["last_purchase"].dt.normalize()).dt.days.clip(lower=0)

    rfm = customers[["customer_id"]].merge(purchase_summary, on="customer_id", how="left")
    rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
    rfm["monetary"] = rfm["monetary"].fillna(0.0)
    rfm["purchase_events"] = rfm["purchase_events"].fillna(0).astype(int)
    rfm["recency"] = rfm["recency"].fillna((as_of_date.normalize() - pd.Timestamp("2015-01-01")).days).clip(lower=0).astype(int)

    rfm["r_score"] = _quantile_score(rfm["recency"], higher_is_better=False)
    rfm["f_score"] = _quantile_score(rfm["frequency"], higher_is_better=True)
    rfm["m_score"] = _quantile_score(rfm["monetary"], higher_is_better=True)
    rfm["rfm_score"] = rfm[["r_score", "f_score", "m_score"]].sum(axis=1)
    return rfm

def segment_customers(customers: pd.DataFrame, events: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    rfm = build_rfm_table(customers, events)
    n_clusters = max(1, min(int(n_clusters), len(rfm)))

    features = rfm[["recency", "frequency", "monetary"]].copy()
    features["frequency"] = np.log1p(features["frequency"])
    features["monetary"] = np.log1p(features["monetary"])
    scaled_features = StandardScaler().fit_transform(features)

    if n_clusters == 1:
        cluster_ids = np.zeros(len(rfm), dtype=int)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        cluster_ids = model.fit_predict(scaled_features)
    rfm["cluster_id"] = cluster_ids

    cluster_summary = (
        rfm.groupby("cluster_id")
        .agg(
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            customers=("customer_id", "nunique"),
        )
        .reset_index()
    )
    cluster_summary["quality_score"] = (
        _standardise(cluster_summary["avg_frequency"])
        + _standardise(cluster_summary["avg_monetary"])
        - _standardise(cluster_summary["avg_recency"])
    )
    ranked_cluster_ids = cluster_summary.sort_values("quality_score", ascending=False)["cluster_id"].tolist()

    segment_names = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Occasional Customers", "Low Engagement"]
    segment_by_cluster = {
        cluster_id: (segment_names[pos] if pos < len(segment_names) else f"Segment {pos + 1}")
        for pos, cluster_id in enumerate(ranked_cluster_ids)
    }
    rfm["segment"] = rfm["cluster_id"].map(segment_by_cluster)
    return rfm

def mine_association_rules(events: pd.DataFrame, products: pd.DataFrame, min_support: float = 0.003, min_confidence: float = 0.15, max_rules: int = 100) -> pd.DataFrame:
    purchases = events.loc[events["event_type"].eq("purchase") & events["order_id"].notna(), ["order_id", "product_id"]].drop_duplicates()
    baskets = purchases.groupby("order_id")["product_id"].apply(set)
    total_baskets = len(baskets)
    output_columns = [
        "rule_id", "antecedent_product_id", "consequent_product_id",
        "support", "confidence", "lift",
        "antecedent_name", "consequent_name"
    ]
    if total_baskets == 0:
        return pd.DataFrame(columns=output_columns)

    item_counts: Counter[str] = Counter()
    pair_counts: Counter[Tuple[str, str]] = Counter()
    for basket in baskets:
        for product_id in basket:
            item_counts[str(product_id)] += 1
        for product_a, product_b in combinations(sorted(map(str, basket)), 2):
            pair_counts[(product_a, product_b)] += 1

    rules: List[dict] = []
    product_lookup = products.set_index("product_id")
    for (product_a, product_b), pair_count in pair_counts.items():
        pair_support = pair_count / total_baskets
        if pair_support < min_support:
            continue
        for antecedent, consequent in ((product_a, product_b), (product_b, product_a)):
            antecedent_support = item_counts[antecedent] / total_baskets
            consequent_support = item_counts[consequent] / total_baskets
            confidence = pair_support / antecedent_support if antecedent_support else 0.0
            lift = confidence / consequent_support if consequent_support else 0.0
            if confidence < min_confidence:
                continue

            antecedent_name = product_lookup.loc[antecedent, "product_name"] if antecedent in product_lookup.index else antecedent
            consequent_name = product_lookup.loc[consequent, "product_name"] if consequent in product_lookup.index else consequent
            rules.append({
                "antecedent_product_id": antecedent,
                "consequent_product_id": consequent,
                "support": round(pair_support, 6),
                "confidence": round(confidence, 6),
                "lift": round(lift, 6),
                "antecedent_name": antecedent_name if isinstance(antecedent_name, str) else str(antecedent_name.iloc[0]),
                "consequent_name": consequent_name if isinstance(consequent_name, str) else str(consequent_name.iloc[0]),
            })

    rules_df = pd.DataFrame(rules)
    if rules_df.empty:
        return pd.DataFrame(columns=output_columns)
    rules_df = rules_df.sort_values(["lift", "confidence", "support"], ascending=False).head(max_rules).reset_index(drop=True)
    rules_df.insert(0, "rule_id", [f"R{idx + 1:04d}" for idx in range(len(rules_df))])
    return rules_df[output_columns]

def _minmax(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = array.min(), array.max()
    if high == low:
        return np.ones_like(array) if high > 0 else np.zeros_like(array)
    return (array - low) / (high - low)

class HybridRecommender:
    def __init__(self, customers: pd.DataFrame, products: pd.DataFrame, segments: pd.DataFrame | None = None) -> None:
        self.customers = customers.copy()
        self.products = products.drop_duplicates(subset=["product_id"]).copy()
        self.segments = segments.copy() if segments is not None else None
        self.item_ids = self.products["product_id"].astype(str).tolist()
        self.item_to_index = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.customer_ids = self.customers["customer_id"].astype(str).tolist()
        self.customer_to_index = {cust_id: idx for idx, cust_id in enumerate(self.customer_ids)}
        self.product_names = self.products.set_index("product_id")["product_name"].astype(str).to_dict()
        self.product_prices = self.products.set_index("product_id")["price"].astype(float).to_dict()

    def _interaction_matrix(self, events: pd.DataFrame) -> csr_matrix:
        weighted = events.loc[events["event_type"].isin(EVENT_WEIGHTS)].copy()
        weighted["weight"] = weighted["event_type"].map(EVENT_WEIGHTS)
        weighted = weighted[
            weighted["customer_id"].isin(self.customer_to_index)
            & weighted["product_id"].isin(self.item_to_index)
        ]
        if weighted.empty:
            return csr_matrix((len(self.customer_ids), len(self.item_ids)), dtype=float)

        rows = weighted["customer_id"].map(self.customer_to_index).to_numpy()
        cols = weighted["product_id"].map(self.item_to_index).to_numpy()
        vals = weighted["weight"].astype(float).to_numpy()
        mat = csr_matrix((vals, (rows, cols)), shape=(len(self.customer_ids), len(self.item_ids)))
        mat.sum_duplicates()
        mat.data = np.log1p(mat.data)
        return mat

    def fit(self, events: pd.DataFrame) -> "HybridRecommender":
        self.events = events.copy()
        self.interaction_matrix = self._interaction_matrix(events)

        purchases = events.loc[
            events["event_type"].eq("purchase")
            & events["customer_id"].isin(self.customer_to_index)
            & events["product_id"].isin(self.item_to_index)
        ].copy()

        if purchases.empty:
            self.purchase_matrix = csr_matrix((len(self.customer_ids), len(self.item_ids)), dtype=float)
            self.item_similarity = np.zeros((len(self.item_ids), len(self.item_ids)))
        else:
            rows = purchases["customer_id"].map(self.customer_to_index).to_numpy()
            cols = purchases["product_id"].map(self.item_to_index).to_numpy()
            vals = purchases["quantity"].astype(float).clip(lower=1).to_numpy()
            p_mat = csr_matrix((vals, (rows, cols)), shape=(len(self.customer_ids), len(self.item_ids)))
            p_mat.sum_duplicates()
            p_mat.data = np.log1p(p_mat.data)
            self.purchase_matrix = p_mat
            self.item_similarity = cosine_similarity(self.purchase_matrix.T, dense_output=True)

        self.global_popularity = (
            purchases.groupby("product_id").size().reindex(self.item_ids, fill_value=0).astype(float).to_numpy()
        )

        if self.segments is not None and not purchases.empty:
            seg_events = purchases.merge(self.segments[["customer_id", "segment"]], on="customer_id", how="left")
            seg_counts = seg_events.groupby(["segment", "product_id"]).size()
            self.segment_popularity = {}
            for seg, seg_frame in seg_counts.groupby(level=0):
                self.segment_popularity[str(seg)] = (
                    seg_frame.droplevel(0).reindex(self.item_ids, fill_value=0).astype(float).to_numpy()
                )
        else:
            self.segment_popularity = {}
        return self

    def _known_products(self, customer_id: str) -> set[str]:
        if customer_id not in self.customer_to_index:
            return set()
        row = self.purchase_matrix.getrow(self.customer_to_index[customer_id])
        return {self.item_ids[idx] for idx in row.indices}

    def recommend(self, customer_id: str, k: int = 10) -> pd.DataFrame:
        customer_id = str(customer_id)
        n_items = len(self.item_ids)
        if n_items == 0:
            return pd.DataFrame(columns=["product_id", "score", "reason"])

        if customer_id in self.customer_to_index:
            u_idx = self.customer_to_index[customer_id]
            inter_scores = self.interaction_matrix.getrow(u_idx).toarray().ravel()
            collab_scores = inter_scores @ self.item_similarity
            seg = None
            if self.segments is not None:
                match = self.segments.loc[self.segments["customer_id"].astype(str).eq(customer_id), "segment"]
                if not match.empty:
                    seg = str(match.iloc[0])

            seg_scores = self.segment_popularity.get(seg, np.zeros(n_items))
            reason = "hybrid_personalised"
        else:
            collab_scores = np.zeros(n_items)
            seg_scores = np.zeros(n_items)
            reason = "popular_cold_start"

        scores = (
            0.65 * _minmax(collab_scores)
            + 0.20 * _minmax(seg_scores)
            + 0.15 * _minmax(self.global_popularity)
        )

        known = self._known_products(customer_id)
        for item_id in known:
            scores[self.item_to_index[item_id]] = -np.inf

        ranking = np.argsort(-scores, kind="stable")
        rows = []
        for idx in ranking:
            if not np.isfinite(scores[idx]):
                continue
            rows.append({"product_id": self.item_ids[int(idx)], "score": round(float(scores[idx]), 6), "reason": reason})
            if len(rows) >= int(k):
                break

        if not rows:
            fallback = np.argsort(-self.global_popularity, kind="stable")
            rows = [{"product_id": self.item_ids[int(idx)], "score": round(float(self.global_popularity[idx]), 6), "reason": "popular_fallback"} for idx in fallback[: int(k)]]
        return pd.DataFrame(rows)

def make_recommendation_table(model: HybridRecommender, customers: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []
    seg_lookup = model.segments.set_index("customer_id")["segment"].astype(str).to_dict() if model.segments is not None else {}
    for cust_id in customers["customer_id"].astype(str):
        rec_df = model.recommend(cust_id, k=k)
        for rank, rec in enumerate(rec_df.to_dict("records"), start=1):
            pid = rec["product_id"]
            rows.append({
                "customer_id": cust_id,
                "segment": seg_lookup.get(cust_id, "Unknown"),
                "rank": rank,
                "product_id": pid,
                "product_name": model.product_names.get(pid, "Unknown"),
                "price": model.product_prices.get(pid, np.nan),
                "score": rec["score"],
                "reason": rec["reason"],
            })
    return pd.DataFrame(rows)

def time_based_holdout(customers: pd.DataFrame, events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    purchases = events.loc[events["event_type"].eq("purchase")].sort_values("timestamp")
    test_rows, cutoffs, test_pairs = [], {}, set()

    for cust_id, cust_purchases in purchases.groupby("customer_id"):
        cust_purchases = cust_purchases.sort_values("timestamp").reset_index(drop=True)
        candidate = None
        for pos in range(len(cust_purchases) - 1, 0, -1):
            prior_pids = set(cust_purchases.loc[: pos - 1, "product_id"].astype(str))
            candidate_pid = str(cust_purchases.loc[pos, "product_id"])
            if candidate_pid not in prior_pids:
                candidate = cust_purchases.loc[pos]
                break
        if candidate is not None:
            c_key = str(cust_id)
            cutoffs[c_key] = pd.Timestamp(candidate["timestamp"])
            test_pairs.add((c_key, str(candidate["product_id"])))
            test_rows.append({"customer_id": c_key, "actual_product_id": str(candidate["product_id"])})

    if not cutoffs:
        return events.copy(), pd.DataFrame(columns=["customer_id", "actual_product_id"])

    cutoff_series = pd.Series(cutoffs)
    ev_cust = events["customer_id"].astype(str)
    ev_time = pd.to_datetime(events["timestamp"])
    temporal_drop = ev_cust.isin(cutoff_series.index) & (ev_time >= ev_cust.map(cutoff_series))
    pair_series = list(zip(events["customer_id"].astype(str), events["product_id"].astype(str)))
    precursor_drop = pd.Series(
        [pair in test_pairs for pair in pair_series],
        index=events.index,
        dtype=bool,
    )
    train_events = events.loc[~(temporal_drop | precursor_drop)].copy()
    train_events = events.loc[~(temporal_drop | precursor_drop)].copy()
    test = pd.DataFrame(test_rows)
    return train_events, test
# Evaluation 
def evaluate_at_k(model: HybridRecommender, test: pd.DataFrame, k: int = 10) -> Dict[str, float | int]:
    if test.empty:
        return {"evaluated_customers": 0, "precision_at_k": 0.0, "recall_at_k": 0.0, "hit_rate_at_k": 0.0, "ndcg_at_k": 0.0}

    precisions, recalls, hits, ndcgs = [], [], [], []
    for row in test.itertuples(index=False):
        recs = model.recommend(str(row.customer_id), k=k)
        rec_ids = recs["product_id"].astype(str).tolist()
        actual_id = str(row.actual_product_id)

        if actual_id in rec_ids:
            rank = rec_ids.index(actual_id) + 1
            hit = 1.0
            ndcg = 1.0 / np.log2(rank + 1)
        else:
            hit = 0.0
            ndcg = 0.0

        hits.append(hit)
        precisions.append(hit / max(k, 1))
        recalls.append(hit)
        ndcgs.append(ndcg)

    return {
        "evaluated_customers": int(len(test)),
        "precision_at_k": round(float(np.mean(precisions)), 6),
        "recall_at_k": round(float(np.mean(recalls)), 6),
        "hit_rate_at_k": round(float(np.mean(hits)), 6),
        "ndcg_at_k": round(float(np.mean(ndcgs)), 6),
    }
# Intelligent scehma Director (schema level specification)
# this is an automated or user-defined mechanism that handles how data is abstracted, generalized, and navigated across different levels of a
#of database schema 

def auto_detect_and_prepare(df: pd.DataFrame):
    """Detects essential e-commerce column roles automatically."""
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}

    def match(patterns: List[str]) -> str | None:
        for p in patterns:
            for clean_col, orig_col in cols.items():
                if p == clean_col or p in clean_col:
                    return orig_col
        return None

    user_col = match(["customerid", "userid", "customeruniqueid", "customer", "user"])
    item_col = match(["stockcode", "productid", "itemid", "asin", "sku", "product"])
    order_col = match(["invoiceno", "invoice", "orderid", "ordernumber", "order"])
    time_col = match(["invoicedate", "orderdate", "timestamp", "purchasedate", "date", "time"])
    name_col = match(["description", "productname", "itemname", "title", "name"])
    price_col = match(["unitprice", "price", "revenue", "amount", "cost"])
    qty_col = match(["quantity", "qty", "count", "items"])

    print("\nðŸ” Column Detection Results:")
    print(f"  User ID Column    : {user_col}")
    print(f"  Product ID Column : {item_col}")
    print(f"  Order ID Column   : {order_col}")
    print(f"  Timestamp Column  : {time_col}")
    print(f"  Product Name      : {name_col}")
    print(f"  Price Column      : {price_col}")

    if not (user_col and item_col and order_col and time_col):
        raise ValueError("Could not automatically map core IDs (User, Product, Order, or Date). Please check CSV headers.")

    # Data standardisation
    df = df.dropna(subset=[user_col, item_col, order_col]).copy()
    
    df["customer_id"] = df[user_col].astype(str)
    df["product_id"] = df[item_col].astype(str)
    df["order_id"] = df[order_col].astype(str)
    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["product_name"] = df[name_col].astype(str).str.strip() if name_col else df["product_id"]
    df["quantity"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(1).astype(int) if qty_col else 1
    df = df[df["quantity"] > 0]
    
    if price_col:
        df["price"] = pd.to_numeric(df[price_col], errors="coerce").fillna(1.0).astype(float)
        df = df[df["price"] > 0]
    else:
        df["price"] = 1.0

    df["revenue"] = df["quantity"] * df["price"]
    df["event_type"] = "purchase"
    df["event_id"] = [f"E{i+1:08d}" for i in range(len(df))]

    # Cap to top 1,000 customers to ensure Colab runs interactively without RAM issues
    top_cust = df["customer_id"].value_counts().head(1000).index
    df = df[df["customer_id"].isin(top_cust)].copy()

    customers = df[["customer_id"]].drop_duplicates().reset_index(drop=True)
    products = df[["product_id", "product_name", "price"]].drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    orders = df.groupby(["order_id", "customer_id"]).agg(
        order_date=("timestamp", "min"),
        order_total=("revenue", "sum"),
        items_count=("quantity", "sum")
    ).reset_index()
    events = df[["event_id", "customer_id", "product_id", "event_type", "timestamp", "quantity", "price", "revenue", "order_id"]].rename(columns={"price": "unit_price"})

    return customers, products, events, orders

#local CSV runner 


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E-commerce behaviour mining and recommendation engine")
    parser.add_argument("--input", default="data/online_retail.csv", help="Path to the transactional CSV file")
    parser.add_argument("--top-customers", type=int, default=1000, help="Maximum customers to keep")
    args = parser.parse_args()

    print(f"Loading data from: {args.input}")
    try:
        raw_df = pd.read_csv(args.input, encoding="utf-8")
    except UnicodeDecodeError:
        raw_df = pd.read_csv(args.input, encoding="ISO-8859-1")

    customers, products, events, orders = auto_detect_and_prepare(raw_df)
    print(f"Processed Dataset: {len(customers):,} Customers | {len(products):,} Products | {len(events):,} Events")

    print("Step 1/4: Running RFM Segmentation (K-Means)...")
    segments = segment_customers(customers, events, n_clusters=4)

    print("Step 2/4: Mining Market Basket Association Rules...")
    rules = mine_association_rules(events, products, min_support=0.004, min_confidence=0.15)
    rules.to_csv("outputs/association_rules.csv", index=False)

    print("Step 3/4: Evaluating Recommender (Time-Based Holdout)...")
    train_events, test = time_based_holdout(customers, events)
    if test.empty:
        print("No eligible holdout records were found; skipping evaluation.")
        metrics = {"evaluated_customers": 0, "precision_at_k": 0.0, "recall_at_k": 0.0, "hit_rate_at_k": 0.0, "ndcg_at_k": 0.0}
    else:
        eval_segments = segment_customers(customers, train_events, n_clusters=4)
        eval_model = HybridRecommender(customers, products, eval_segments).fit(train_events)
        metrics = evaluate_at_k(eval_model, test, k=10)

    print("Step 4/4: Generating Final Personalized Recommendations...")
    final_model = HybridRecommender(customers, products, segments).fit(events)
    recommendations = make_recommendation_table(final_model, customers, k=10)
    recommendations.to_csv("outputs/recommendations.csv", index=False)
    segments.to_csv("outputs/customer_segments.csv", index=False)

    with open("outputs/evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\\nPROJECT EVALUATION SUMMARY")
    print(f"Frequent Rules Discovered: {len(rules):,}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    if not recommendations.empty:
        sample_user = str(customers.iloc[0]["customer_id"])
        print(f"\\nTop recommendations for customer {sample_user}:")
        print(recommendations[recommendations["customer_id"].astype(str).eq(sample_user)].head(10).to_string(index=False))