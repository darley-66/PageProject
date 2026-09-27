import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

_kmeans_results = None


def _require_results():
    if _kmeans_results is None:
        raise RuntimeError("Call run_kmeans() before requesting K-Means results.")
    return _kmeans_results


def run_kmeans():
    global _kmeans_results

    base = Path(__file__).resolve().parent.parent
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
    iteration_assignments = {}
    iteration_distances = {}
    generated_images = {}

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
    initial_image = output / "initial.png"
    plt.savefig(initial_image, dpi=180)
    plt.close()
    generated_images["initial"] = str(initial_image)

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
        distance_table.insert(0, "customer_id", df["customer_id"])
        distance_table["cluster"] = clusters + 1
        distance_table.to_csv(
            output / f"iteration_{iteration}_distances.csv",
            index=False
        )
        iteration_distances[iteration] = distance_table.copy()

        assignments = df.copy()
        assignments["cluster"] = clusters + 1
        assignments.to_csv(
            output / f"iteration_{iteration}_assignments.csv",
            index=False
        )
        iteration_assignments[iteration] = assignments.copy()

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
            sse = ((points - new_centroids[i]) ** 2).sum()
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
            points = assignments[assignments["cluster"] == i + 1]
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
        iteration_image = output / f"iteration_{iteration}.png"
        plt.savefig(iteration_image, dpi=180)
        plt.close()
        generated_images[f"iteration_{iteration}"] = str(iteration_image)

        centroids = new_centroids

    variance = pd.concat(variances, ignore_index=True)
    variance.to_csv(output / "variance_by_iteration.csv", index=False)

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
    centroids_final.to_csv(output / "centroids.csv", index=False)

    final_distances = np.sqrt(
        ((X[:, None] - centroids) ** 2).sum(axis=2)
    )
    final_clusters = np.argmin(final_distances, axis=1)
    final_assignments = df.copy()
    final_assignments["cluster"] = final_clusters + 1
    final_assignments.to_csv(output / "final_assignments.csv", index=False)

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
    cluster_summary.to_csv(output / "cluster_summary.csv", index=False)

    _kmeans_results = {
        "centroids": centroids_final,
        "cluster_summary": cluster_summary,
        "final_assignments": final_assignments,
        "iteration_assignments": iteration_assignments,
        "iteration_distances": iteration_distances,
        "generated_images": generated_images,
        "variance_comparison": variance,
    }

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

    return {
        "centroids": get_centroids(),
        "cluster_summary": get_cluster_summary(),
        "assignments": get_assignments(),
        "generated_images": get_generated_images(),
        "variance_comparison": get_variance_comparison(),
    }


def get_centroids():
    return _require_results()["centroids"].copy()


def get_cluster_summary():
    return _require_results()["cluster_summary"].copy()


def get_assignments():
    results = _require_results()
    return {
        "final": results["final_assignments"].copy(),
        "iterations": {
            iteration: assignments.copy()
            for iteration, assignments in results["iteration_assignments"].items()
        },
        "distances": {
            iteration: distances.copy()
            for iteration, distances in results["iteration_distances"].items()
        },
    }


def get_generated_images():
    return _require_results()["generated_images"].copy()


def get_variance_comparison():
    return _require_results()["variance_comparison"].copy()