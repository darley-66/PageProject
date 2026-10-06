from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


FEATURE_COLS = ["PurchaseFrequency", "TotalSpending", "AverageSpending"]


def getData():
    data_path = Path(__file__).resolve().parent / "data" / "Online Retail.xlsx"
    transactions = pd.read_excel(
        data_path,
        sheet_name="Online Retail",
        usecols=["CustomerID", "InvoiceNo", "Quantity", "UnitPrice"],
    )

    transactions["CustomerID"] = pd.to_numeric(
        transactions["CustomerID"], errors="coerce"
    )
    transactions = transactions.dropna(subset=["CustomerID"]).copy()
    transactions["CustomerID"] = transactions["CustomerID"].astype("int64")
    transactions["InvoiceNo"] = transactions["InvoiceNo"].astype("string").str.strip()
    transactions["Quantity"] = pd.to_numeric(
        transactions["Quantity"], errors="coerce"
    )
    transactions["UnitPrice"] = pd.to_numeric(
        transactions["UnitPrice"], errors="coerce"
    )
    transactions = transactions.dropna(
        subset=["InvoiceNo", "Quantity", "UnitPrice"]
    )
    transactions = transactions[
        ~transactions["InvoiceNo"].str.upper().str.startswith("C", na=False)
        & (transactions["Quantity"] > 0)
        & (transactions["UnitPrice"] > 0)
    ].copy()

    transactions["LineSpending"] = (
        transactions["Quantity"] * transactions["UnitPrice"]
    )
    invoice_spending = (
        transactions.groupby(["CustomerID", "InvoiceNo"], as_index=False)
        .agg(InvoiceSpending=("LineSpending", "sum"))
    )
    customers = (
        invoice_spending.groupby("CustomerID", as_index=False)
        .agg(
            PurchaseFrequency=("InvoiceNo", "nunique"),
            TotalSpending=("InvoiceSpending", "sum"),
            AverageSpending=("InvoiceSpending", "mean"),
        )
    )

    feature_cols = FEATURE_COLS.copy()
    data = customers[["CustomerID", *feature_cols]].to_dict(orient="records")
    return data, customers, feature_cols

@lru_cache(maxsize=1)
def implementClustering():
    data, df, feature_cols = getData()

    if len(df) < 3:
        raise ValueError("At least three customers are required for K-Means.")

    X = df[feature_cols].to_numpy(dtype=float)
    X_transformed = X.copy()
    for feature in ("TotalSpending", "AverageSpending"):
        feature_index = feature_cols.index(feature)
        X_transformed[:, feature_index] = np.log1p(X[:, feature_index])

    scaler = StandardScaler()
    Xscaled = scaler.fit_transform(X_transformed)

    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(Xscaled)

    result = []
    for row_data, label in zip(data, labels):
        row = row_data.copy()
        row["cluster"] = int(label)
        result.append(row)

    summary_clusters = {}
    for label in labels:
        label = int(label)
        summary_clusters[label] = summary_clusters.get(label, 0) + 1

    centroids_transformed = scaler.inverse_transform(model.cluster_centers_)
    for feature in ("TotalSpending", "AverageSpending"):
        feature_index = feature_cols.index(feature)
        centroids_transformed[:, feature_index] = np.expm1(
            centroids_transformed[:, feature_index]
        )
    centroids = centroids_transformed.tolist()

    cluster_count = len(set(labels))
    score = (
        silhouette_score(Xscaled, labels)
        if 1 < cluster_count < len(labels)
        else None
    )

    plot_data = pd.DataFrame(result)
    figure, axis = plt.subplots(figsize=(10, 6))
    for cluster_id, cluster_rows in plot_data.groupby("cluster", sort=True):
        axis.scatter(
            cluster_rows["PurchaseFrequency"],
            cluster_rows["TotalSpending"],
            label=f"Cluster {cluster_id}",
            alpha=0.65,
            s=28,
        )

    frequency_index = feature_cols.index("PurchaseFrequency")
    spending_index = feature_cols.index("TotalSpending")
    centroid_frequencies = [centroid[frequency_index] for centroid in centroids]
    centroid_spending = [centroid[spending_index] for centroid in centroids]
    axis.scatter(
        centroid_frequencies,
        centroid_spending,
        marker="X",
        s=320,
        c="#111827",
        edgecolors="#facc15",
        linewidths=1.8,
        label="Centroids",
        zorder=5,
    )
    for cluster_id, (frequency, spending) in enumerate(
        zip(centroid_frequencies, centroid_spending)
    ):
        axis.annotate(
            f"C{cluster_id}",
            (frequency, spending),
            xytext=(7, 6),
            textcoords="offset points",
            fontweight="bold",
        )

    axis.set_yscale("log")
    axis.set_xlabel("Purchase Frequency")
    axis.set_ylabel("Total Spending (log scale)")
    axis.set_title("Customer Clusters")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend()
    figure.tight_layout()

    project_dir = Path(__file__).resolve().parent
    output_path = project_dir / "outputs" / "kmeans_clusters.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    plot_path = output_path.relative_to(project_dir).as_posix()

    return {
        "results": result,
        "summary": summary_clusters,
        "centroids": centroids,
        "silhouette_score": score,
        "feature_cols": feature_cols,
        "plot_path": plot_path,
    }