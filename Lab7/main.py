import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

from report_generator import generate_lab7_html_report

def find_file(relative_paths):
    """Locate file from multiple candidate paths."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    for rel in relative_paths:
        cand_base = os.path.join(base_dir, rel)
        if os.path.exists(cand_base):
            return os.path.abspath(cand_base)
        cand_cwd = os.path.join(cwd, rel)
        if os.path.exists(cand_cwd):
            return os.path.abspath(cand_cwd)
    return None

def find_default_dataset():
    """Find student dataset with fallback options."""
    candidates = [
        "../DataSet/student_dataset_full_normalized.csv",
        "DataSet/student_dataset_full_normalized.csv",
        "../DataSet/student_performance_normalized.csv",
        "DataSet/student_performance_normalized.csv",
        "student_dataset_full_normalized.csv"
    ]
    path = find_file(candidates)
    if path:
        return path
    raise FileNotFoundError("Could not find suitable student dataset in DataSet/ folder.")

def evaluate_k_range(X: np.ndarray, k_min: int = 2, k_max: int = 8, random_state: int = 42):
    """
    Evaluate K-Means across a range of K values using Inertia (WCSS) and Silhouette Score.
    """
    k_values = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = []
    models = {}

    print(f"\n{'='*65}")
    print(f" 🔍 EVALUATING CLUSTER QUALITY (K = {k_min} TO {k_max})")
    print(f"{'='*65}")
    print(f"{'K':<5} | {'Inertia (WCSS)':<18} | {'Silhouette Score':<18} | {'Status'}")
    print("-" * 65)

    best_k = k_min
    best_sil = -1.0

    # Subsample if dataset is very large to speed up silhouette computation
    if len(X) > 5000:
        sample_indices = np.random.RandomState(random_state).choice(len(X), size=5000, replace=False)
        X_eval = X[sample_indices]
    else:
        sample_indices = None
        X_eval = X

    for k in k_values:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=random_state)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))
        
        # Silhouette score
        eval_labels = labels[sample_indices] if sample_indices is not None else labels
        sil = float(silhouette_score(X_eval, eval_labels))
        silhouettes.append(sil)
        models[k] = (km, labels)

        status_tag = ""
        if sil > best_sil:
            best_sil = sil
            best_k = k
            status_tag = "⭐ Best Silhouette"

        print(f"{k:<5} | {km.inertia_:>18,.2f} | {sil:>18.4f} | {status_tag}")

    print("-" * 65)
    print(f"💡 Recommended Optimal K based on Silhouette Score: K = {best_k} (Score: {best_sil:.4f})")
    
    return {
        'k_values': k_values,
        'inertias': inertias,
        'silhouettes': silhouettes,
        'best_k': best_k,
        'models': models
    }

def save_matplotlib_figures(elbow_data, pca_data, output_dir):
    """Save static PNG figures for presentations and reports."""
    # 1. Elbow & Silhouette Curves
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color_inertia = 'tab:orange'
    ax1.set_xlabel('Number of Clusters (K)', fontweight='bold')
    ax1.set_ylabel('Inertia / WCSS', color=color_inertia, fontweight='bold')
    line1 = ax1.plot(elbow_data['k_values'], elbow_data['inertias'], marker='o', color=color_inertia, linewidth=2, label='Inertia (WCSS)')
    ax1.tick_params(axis='y', labelcolor=color_inertia)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color_sil = 'tab:purple'
    ax2.set_ylabel('Silhouette Score', color=color_sil, fontweight='bold')
    line2 = ax2.plot(elbow_data['k_values'], elbow_data['silhouettes'], marker='s', color=color_sil, linewidth=2, linestyle='--', label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color_sil)

    plt.title('Elbow Method & Silhouette Score vs. Number of Clusters', fontsize=12, fontweight='bold', pad=15)
    fig.tight_layout()
    elbow_path = os.path.join(output_dir, 'lab7_elbow_analysis.png')
    plt.savefig(elbow_path, dpi=300)
    plt.close()

    # 2. PCA 2D Cluster Scatter Plot
    plt.figure(figsize=(9, 6))
    clusters = sorted(list(set(p['cluster'] for p in pca_data['points'])))
    colors = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444', '#06b6d4', '#6366f1', '#f97316']

    for c in clusters:
        pts = [p for p in pca_data['points'] if p['cluster'] == c]
        xs = [p['x'] for p in pts]
        ys = [p['y'] for p in pts]
        plt.scatter(xs, ys, color=colors[c % len(colors)], alpha=0.6, s=25, label=f'Cluster {c}')

    # Centroids
    cx = [c['x'] for c in pca_data['centroids']]
    cy = [c['y'] for c in pca_data['centroids']]
    plt.scatter(cx, cy, color='black', marker='X', s=180, edgecolors='white', linewidths=1.5, label='Centroids', zorder=5)

    plt.title(f'K-Means Clusters in 2D PCA Space (K={len(clusters)})', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Principal Component 1 (PC1)', fontweight='bold')
    plt.ylabel('Principal Component 2 (PC2)', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, loc='best')
    plt.tight_layout()
    pca_plot_path = os.path.join(output_dir, 'lab7_pca_clusters.png')
    plt.savefig(pca_plot_path, dpi=300)
    plt.close()

    return elbow_path, pca_plot_path

def main():
    parser = argparse.ArgumentParser(description="K-Means Clustering on N-Features Dataset (AI Lab 7)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--features", type=str, default=None, help="Comma-separated feature column names (default: all numeric)")
    parser.add_argument("--scale", type=str, default="standard", choices=["standard", "minmax", "none"], help="Feature scaling method")
    parser.add_argument("--k", type=str, default="auto", help="Number of clusters K (integer or 'auto')")
    parser.add_argument("--k-min", type=int, default=2, help="Minimum K for evaluation")
    parser.add_argument("--k-max", type=int, default=8, help="Maximum K for evaluation")
    parser.add_argument("--save-plots", action="store_true", default=True, help="Save PNG plots using matplotlib")
    parser.add_argument("--no-report", action="store_true", help="Skip generating HTML report")
    
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 65)
    print(" LAB 7: K-MEANS CLUSTERING (N-DIMENSIONAL FEATURES)")
    print("=" * 65)

    # 1. Dataset Loading
    dataset_path = args.dataset if args.dataset else find_default_dataset()
    print(f"📂 Dataset Loaded: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"📊 Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # 2. Dynamic Feature Selection (any number of features)
    if args.features:
        features = [f.strip() for f in args.features.split(",") if f.strip() in df.columns]
        if not features:
            print("⚠️ Warning: None of the specified features found. Falling back to numeric features.")
            features = list(df.select_dtypes(include=[np.number]).columns)
    else:
        features = list(df.select_dtypes(include=[np.number]).columns)
        # Drop identifier columns if any (like StudentID, id)
        features = [f for f in features if not f.lower().endswith("id")]

    print(f"\n🔢 Selected {len(features)} Features for Clustering:")
    for idx, f in enumerate(features, 1):
        print(f"   [{idx}] {f} (Min: {df[f].min():.2f}, Max: {df[f].max():.2f}, Mean: {df[f].mean():.2f})")

    # Prepare feature matrix
    X_raw_df = df[features].dropna()
    X_raw = X_raw_df.values

    # Optional Scaling to ensure fair contribution across all N features
    scaler = None
    if args.scale == "standard":
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        print(f"\n⚙️ Feature Scaling: StandardScaler applied (Zero mean, unit variance for all {len(features)} features)")
    elif args.scale == "minmax":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X_raw)
        print(f"\n⚙️ Feature Scaling: MinMaxScaler applied ([0, 1] range for all {len(features)} features)")
    else:
        X = X_raw
        print("\n⚙️ Feature Scaling: None (Using raw feature values)")

    # 3. K Evaluation (Elbow & Silhouette)
    eval_res = evaluate_k_range(X, k_min=args.k_min, k_max=args.k_max)

    # Determine final K
    if args.k.lower() == "auto":
        k_final = eval_res['best_k']
        print(f"\n🎯 Automatically selected optimal K = {k_final}")
    else:
        try:
            k_final = int(args.k)
            print(f"\n🎯 User selected K = {k_final}")
        except ValueError:
            k_final = eval_res['best_k']
            print(f"\n⚠️ Invalid K '{args.k}', default to optimal K = {k_final}")

    # 4. Final Model Training
    if k_final in eval_res['models']:
        final_model, labels = eval_res['models'][k_final]
    else:
        final_model = KMeans(n_clusters=k_final, init='k-means++', n_init=10, max_iter=300, random_state=42)
        labels = final_model.fit_predict(X)

    final_inertia = float(final_model.inertia_)
    sample_size = min(5000, len(X))
    final_sil = float(silhouette_score(X[:sample_size], labels[:sample_size]))

    # Un-scale centroids for interpretable reporting if scaling was used
    if scaler is not None:
        unscaled_centroids = scaler.inverse_transform(final_model.cluster_centers_)
    else:
        unscaled_centroids = final_model.cluster_centers_

    # 5. Cluster Profiling and Statistics
    df_result = X_raw_df.copy()
    df_result['Cluster'] = labels
    
    cluster_stats = []
    print(f"\n{'='*65}")
    print(f" 📌 CLUSTER PROFILING SUMMARY (K = {k_final})")
    print(f"{'='*65}")

    for c in range(k_final):
        sub_c = df_result[df_result['Cluster'] == c]
        count = len(sub_c)
        pct = (count / len(df_result)) * 100.0
        centroid_dict = {f: float(unscaled_centroids[c, idx]) for idx, f in enumerate(features)}
        
        cluster_stats.append({
            'cluster_id': c,
            'count': count,
            'pct': pct,
            'centroid': centroid_dict
        })

        print(f"\n🔵 Cluster {c}: {count:,} samples ({pct:.2f}%)")
        print("   Centroid Coordinates (Original Scale):")
        for f in features:
            print(f"     • {f:<25}: {centroid_dict[f]:.4f}")

    # 6. PCA 2D Dimensionality Reduction for Visualization
    print(f"\n{'='*65}")
    print(" 📉 DIMENSIONALITY REDUCTION (PCA 2D PROJECTION)")
    print(f"{'='*65}")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    centroids_pca = pca.transform(final_model.cluster_centers_)

    var_ratio = [float(r) for r in pca.explained_variance_ratio_]
    print(f"PCA Component 1 Variance Explained: {var_ratio[0]*100:.2f}%")
    print(f"PCA Component 2 Variance Explained: {var_ratio[1]*100:.2f}%")
    print(f"Total 2D Variance Captured: {(var_ratio[0] + var_ratio[1])*100:.2f}%")

    # Sample points for lightweight frontend rendering (up to 1500 points)
    n_display = min(1500, len(X))
    sub_indices = np.random.RandomState(42).choice(len(X), size=n_display, replace=False) if len(X) > n_display else np.arange(len(X))
    
    pca_data = {
        'points': [
            {'x': float(X_pca[i, 0]), 'y': float(X_pca[i, 1]), 'cluster': int(labels[i])}
            for i in sub_indices
        ],
        'centroids': [
            {'x': float(centroids_pca[c, 0]), 'y': float(centroids_pca[c, 1]), 'cluster': c}
            for c in range(k_final)
        ]
    }

    # 7. Static Plots
    if args.save_plots:
        elbow_img, pca_img = save_matplotlib_figures(
            elbow_data={'k_values': eval_res['k_values'], 'inertias': eval_res['inertias'], 'silhouettes': eval_res['silhouettes']},
            pca_data=pca_data,
            output_dir=base_dir
        )
        print(f"\n🖼️ Saved static plots:")
        print(f"   - {elbow_img}")
        print(f"   - {pca_img}")

    # 8. HTML Report
    if not args.no_report:
        summary_data = {
            'dataset_name': os.path.basename(dataset_path),
            'features': features,
            'k_selected': k_final,
            'total_samples': len(X),
            'inertia': final_inertia,
            'silhouette': final_sil,
            'iterations': int(final_model.n_iter_),
            'pca_explained_variance': var_ratio
        }

        elbow_data = {
            'k_values': eval_res['k_values'],
            'inertias': [round(x, 2) for x in eval_res['inertias']],
            'silhouettes': [round(x, 4) for x in eval_res['silhouettes']]
        }

        report_path = os.path.join(base_dir, "lab7_report.html")
        generate_lab7_html_report(summary_data, elbow_data, cluster_stats, pca_data, output_path=report_path)
        print(f"\n🌐 Interactive HTML Report generated:")
        print(f"   👉 {report_path}")

    print("\n✅ K-Means Clustering Analysis Completed Successfully!")

if __name__ == "__main__":
    main()
