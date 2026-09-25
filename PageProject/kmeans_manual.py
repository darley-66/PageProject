import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).resolve().parent
data = base / "data" / "ecommerce_customers.csv"
output = base / "outputs"

output.mkdir(exist_ok=True)

df = pd.read_csv(data)

X = df[["purchase_frequency", "monthly_spending"]].values.astype(float)

centroids = np.array([
    [2, 100],
    [6, 300],
    [10, 500]
], dtype=float)

history = [centroids.copy()]
variances = []

plt.figure(figsize=(9, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    label="Customers"
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=180,
    label="Initial Centroids"
)

plt.xlabel("Purchase Frequency")
plt.ylabel("Monthly Spending (thousand COP)")
plt.title("Initial Clusters")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    output / "initial.png",
    dpi=180
)

plt.close()

for iteration in range(1, 4):

    distances = np.sqrt(
        ((X[:, None] - centroids) ** 2).sum(axis=2)
    )

    clusters = np.argmin(distances, axis=1)

    distance_table = pd.DataFrame(
        distances,
        columns=[
            "distance_C1",
            "distance_C2",
            "distance_C3"
        ]
    )

    distance_table.insert(
        0,
        "customer_id",
        df["customer_id"]
    )

    distance_table["cluster"] = clusters + 1

    distance_table.to_csv(
        output / f"iteration_{iteration}_distances.csv",
        index=False
    )

    assignments = df.copy()
    assignments["cluster"] = clusters + 1

    assignments.to_csv(
        output / f"iteration_{iteration}_assignments.csv",
        index=False
    )

    new_centroids = []

    for i in range(3):
        points = X[clusters == i]
        new_centroids.append(points.mean(axis=0))

    new_centroids = np.array(new_centroids)

    history.append(new_centroids.copy())

    rows = []

    for i in range(3):

        points = X[clusters == i]

        frequency_variance = np.var(points[:, 0])
        spending_variance = np.var(points[:, 1])

        sse = (
            (points - new_centroids[i]) ** 2
        ).sum()

        rows.append([
            iteration,
            i + 1,
            len(points),
            frequency_variance,
            spending_variance,
            sse
        ])

    variances.append(
        pd.DataFrame(
            rows,
            columns=[
                "iteration",
                "cluster",
                "customers",
                "frequency_variance",
                "spending_variance",
                "SSE"
            ]
        )
    )

    plt.figure(figsize=(9, 6))

    for i in range(3):

        points = assignments[
            assignments["cluster"] == i + 1
        ]

        plt.scatter(
            points["purchase_frequency"],
            points["monthly_spending"],
            label=f"Cluster {i + 1}"
        )

    plt.scatter(
        new_centroids[:, 0],
        new_centroids[:, 1],
        marker="X",
        s=180,
        label="Centroids"
    )

    plt.xlabel("Purchase Frequency")
    plt.ylabel("Monthly Spending (thousand COP)")
    plt.title(f"Iteration {iteration}")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        output / f"iteration_{iteration}.png",
        dpi=180
    )

    plt.close()

    centroids = new_centroids

variance = pd.concat(
    variances,
    ignore_index=True
)

variance.to_csv(
    output / "variance_by_iteration.csv",
    index=False
)

centroid_rows = []

for i, group in enumerate(history):

    stage = "Initial" if i == 0 else f"Iteration {i}"

    for j, centroid in enumerate(group):

        centroid_rows.append([
            stage,
            j + 1,
            centroid[0],
            centroid[1]
        ])

centroids_final = pd.DataFrame(
    centroid_rows,
    columns=[
        "stage",
        "centroid",
        "purchase_frequency",
        "monthly_spending"
    ]
)

centroids_final.to_csv(
    output / "centroids.csv",
    index=False
)

final_distances = np.sqrt(
    ((X[:, None] - centroids) ** 2).sum(axis=2)
)

final_clusters = np.argmin(
    final_distances,
    axis=1
)

final_assignments = df.copy()
final_assignments["cluster"] = final_clusters + 1

final_assignments.to_csv(
    output / "final_assignments.csv",
    index=False
)

cluster_summary = (
    final_assignments
    .groupby("cluster")
    .agg(
        customers=("customer_id", "count"),
        average_purchase_frequency=(
            "purchase_frequency",
            "mean"
        ),
        average_monthly_spending=(
            "monthly_spending",
            "mean"
        )
    )
    .reset_index()
)

cluster_summary.to_csv(
    output / "cluster_summary.csv",
    index=False
)

print("Process completed")
print()
print("Final centroids:")

for i, centroid in enumerate(centroids):

    print(
        f"C{i + 1}: "
        f"({centroid[0]:.2f}, "
        f"{centroid[1]:.2f})"
    )

print()
print("Final cluster summary:")
print(cluster_summary.to_string(index=False))

print()
print("Files generated in:")
print(output)