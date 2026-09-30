import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from report_generator import generate_lab5_html_report

def find_file(relative_paths):
    """Locate file from multiple potential relative paths."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    for rel in relative_paths:
        candidate = os.path.join(base_dir, rel)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        candidate_cwd = os.path.join(cwd, rel)
        if os.path.exists(candidate_cwd):
            return os.path.abspath(candidate_cwd)
    return None

def format_cm(cm, classes):
    """Format confusion matrix as a neat DataFrame string."""
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in classes],
        columns=[f"Pred {c}" for c in classes]
    )
    return cm_df

def run_5fold_cross_validation(model_name, model_instance, X, y, classes, skf):
    """
    Execute 5-Fold Cross Validation (20% validation split per fold),
    displaying Confusion Matrix (Actual Table) and Metrics per fold.
    """
    print("\n" + "#" * 78)
    print(f" ALGORITHM: {model_name.upper()}")
    print("#" * 78)

    fold_results = []
    fold_details = []

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_val)

        # 1. Confusion Matrix (Actual Table)
        cm = confusion_matrix(y_val, y_pred, labels=classes)
        cm_df = format_cm(cm, classes)

        # 2. Performance Metrics
        acc = accuracy_score(y_val, y_pred)
        prec_weighted = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        rec_weighted = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        prec_macro = precision_score(y_val, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_val, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)

        # Per-class metrics
        prec_per_class = precision_score(y_val, y_pred, labels=classes, average=None, zero_division=0)
        rec_per_class = recall_score(y_val, y_pred, labels=classes, average=None, zero_division=0)
        f1_per_class = f1_score(y_val, y_pred, labels=classes, average=None, zero_division=0)

        print("\n" + "=" * 78)
        print(f" >>> {model_name} - FOLD {fold_num} / 5 (Validation Data: 20% = {len(val_idx):,} rows) <<<")
        print("=" * 78)

        print("\n[1] Actual Table (Confusion Matrix):")
        print(cm_df.to_string())

        print("\n[2] Per-Class Performance Metrics:")
        class_metrics_records = []
        for idx, c in enumerate(classes):
            supp = int(np.sum(y_val == c))
            class_metrics_records.append({
                'Class': c,
                'Precision (%)': float(np.round(prec_per_class[idx] * 100, 2)),
                'Recall (%)': float(np.round(rec_per_class[idx] * 100, 2)),
                'F-Measure (%)': float(np.round(f1_per_class[idx] * 100, 2)),
                'Support': supp
            })
        class_metrics_df = pd.DataFrame(class_metrics_records)
        print(class_metrics_df.to_string(index=False))

        print("\n[3] Fold Evaluation Summary:")
        print(f"   * Accuracy (Overall)    : {acc * 100:.2f}%")
        print(f"   * Precision (Weighted) : {prec_weighted * 100:.2f}%  |  (Macro): {prec_macro * 100:.2f}%")
        print(f"   * Recall (Weighted)    : {rec_weighted * 100:.2f}%  |  (Macro): {rec_macro * 100:.2f}%")
        print(f"   * F-Measure (Weighted) : {f1_weighted * 100:.2f}%  |  (Macro): {f1_macro * 100:.2f}%")

        fold_results.append({
            'Fold': f"Fold {fold_num}",
            'Accuracy': acc * 100,
            'Precision': prec_weighted * 100,
            'Recall': rec_weighted * 100,
            'F-Measure': f1_weighted * 100
        })

        fold_details.append({
            'fold_num': fold_num,
            'val_samples': len(val_idx),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': class_metrics_records,
            'accuracy': round(float(acc * 100), 2),
            'prec_weighted': round(float(prec_weighted * 100), 2),
            'rec_weighted': round(float(rec_weighted * 100), 2),
            'f1_weighted': round(float(f1_weighted * 100), 2),
            'prec_macro': round(float(prec_macro * 100), 2),
            'rec_macro': round(float(rec_macro * 100), 2),
            'f1_macro': round(float(f1_macro * 100), 2)
        })

    # Summary across 5 Folds for this algorithm
    summary_df = pd.DataFrame(fold_results)
    
    mean_row = {
        'Fold': 'Mean (Average)',
        'Accuracy': float(summary_df['Accuracy'].mean()),
        'Precision': float(summary_df['Precision'].mean()),
        'Recall': float(summary_df['Recall'].mean()),
        'F-Measure': float(summary_df['F-Measure'].mean())
    }
    std_row = {
        'Fold': 'Std (Deviation)',
        'Accuracy': float(summary_df['Accuracy'].std()),
        'Precision': float(summary_df['Precision'].std()),
        'Recall': float(summary_df['Recall'].std()),
        'F-Measure': float(summary_df['F-Measure'].std())
    }
    
    summary_with_stats = pd.concat([summary_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    print("\n" + "-" * 78)
    print(f" 5-Fold Cross Validation Summary for {model_name}")
    print("-" * 78)
    formatted_summary = summary_with_stats.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F-Measure']:
        formatted_summary[col] = formatted_summary[col].apply(lambda x: f"{x:.2f}%")
    print(formatted_summary.to_string(index=False))
    print("-" * 78)

    return {
        'Model': model_name,
        'Accuracy_mean': float(summary_df['Accuracy'].mean()),
        'Accuracy_std': float(summary_df['Accuracy'].std()),
        'Precision_mean': float(summary_df['Precision'].mean()),
        'Precision_std': float(summary_df['Precision'].std()),
        'Recall_mean': float(summary_df['Recall'].mean()),
        'Recall_std': float(summary_df['Recall'].std()),
        'F-Measure_mean': float(summary_df['F-Measure'].mean()),
        'F-Measure_std': float(summary_df['F-Measure'].std()),
        'fold_results': fold_results,
        'summary_rows': summary_with_stats.to_dict('records'),
        'fold_details': fold_details
    }

def main():
    print("=" * 78)
    print(" LAB 5: MODEL EVALUATION & COMPARISON (20% CROSS VALIDATION)")
    print(" Comparing Performance: Naive Bayes vs Decision Tree vs K-NN")
    print("=" * 78)

    # 1. Load dataset (Prioritizing the 10,000-row full normalized dataset)
    dataset_path = find_file([
        '../DataSet/student_dataset_full_normalized.csv',
        'DataSet/student_dataset_full_normalized.csv',
        '../DataSet/student_performance_normalized.csv',
        'DataSet/student_performance_normalized.csv'
    ])

    if not dataset_path:
        print("[*] Normalized dataset not found, generating via Lab3 normalize pipeline...")
        try:
            lab3_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Lab3'))
            if lab3_dir not in sys.path:
                sys.path.append(lab3_dir)
            from normalize import normalize_10k_dataset
            normalize_10k_dataset()
            dataset_path = find_file(['../DataSet/student_dataset_full_normalized.csv', 'DataSet/student_dataset_full_normalized.csv'])
        except Exception as err:
            print(f"[!] Warning: Auto-normalization attempt failed: {err}")

    if not dataset_path or not os.path.exists(dataset_path):
        print("[!] Error: Could not locate dataset in DataSet directory.")
        sys.exit(1)

    print(f"\n[1] Loading dataset: {os.path.basename(dataset_path)}")
    df = pd.read_csv(dataset_path)
    target_col = 'FinalGrade' if 'FinalGrade' in df.columns else df.columns[-1]

    drop_cols = [c for c in ['StudentID', 'Name', 'exam_score', 'placement_status'] if c in df.columns and c != target_col]
    df_clean = df.drop(columns=drop_cols).dropna()

    X = df_clean.drop(columns=[target_col]).copy()
    y = df_clean[target_col].astype(str)

    # Label encoding for categorical columns
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype).startswith('str'):
            le = LabelEncoder()
            if col == 'PreviousGrade':
                le.fit(['A', 'B', 'C', 'D', 'F'])
                X[col] = le.transform(X[col].astype(str))
            else:
                X[col] = le.fit_transform(X[col].astype(str))

    X_mat = X.astype(float).values
    y_vec = y.values
    classes = sorted(list(np.unique(y_vec)))

    print(f"  * Total Samples        : {len(df_clean):,} rows")
    print(f"  * Features ({len(X.columns)})       : {', '.join(X.columns)}")
    print(f"  * Target Classes       : {', '.join(classes)}")
    print(f"  * Validation Scheme    : 5-Fold Stratified Cross Validation (20% Holdout per Fold)")

    # 2. Setup 5-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Setup models from Labs 2, 3, and 4
    models = {
        'Naive Bayes (Bayes)': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors (K-NN, K=5)': KNeighborsClassifier(n_neighbors=5)
    }

    all_model_stats = []

    # 4. Run Cross Validation for each algorithm
    for name, model in models.items():
        stats = run_5fold_cross_validation(name, model, X_mat, y_vec, classes, skf)
        all_model_stats.append(stats)

    # 5. Final Comparison Table across all 3 models
    print("\n" + "=" * 78)
    print(" FINAL COMPARISON TABLE (ACROSS 3 ALGORITHMS)")
    print("=" * 78)

    comparison_data = []
    for s in all_model_stats:
        comparison_data.append({
            'Model': s['Model'],
            'Accuracy (%)': f"{s['Accuracy_mean']:.2f} ± {s['Accuracy_std']:.2f}",
            'Precision (%)': f"{s['Precision_mean']:.2f} ± {s['Precision_std']:.2f}",
            'Recall (%)': f"{s['Recall_mean']:.2f} ± {s['Recall_std']:.2f}",
            'F-Measure (%)': f"{s['F-Measure_mean']:.2f} ± {s['F-Measure_std']:.2f}"
        })

    comp_df = pd.DataFrame(comparison_data)
    print(comp_df.to_string(index=False))
    print("=" * 78)

    # 6. Conclusion
    best_acc_model = max(all_model_stats, key=lambda x: x['Accuracy_mean'])
    best_f1_model = max(all_model_stats, key=lambda x: x['F-Measure_mean'])
    best_prec_model = max(all_model_stats, key=lambda x: x['Precision_mean'])

    print("\n[ Summary Conclusions ]:")
    print(f" 1. Highest Mean Accuracy    : {best_acc_model['Model']} ({best_acc_model['Accuracy_mean']:.2f}%)")
    print(f" 2. Highest Mean F-Measure  : {best_f1_model['Model']} ({best_f1_model['F-Measure_mean']:.2f}%)")
    print(f" 3. Highest Mean Precision  : {best_prec_model['Model']} ({best_prec_model['Precision_mean']:.2f}%)")
    print("=" * 78 + "\n")

    # 7. Generate clean simplified HTML report
    dataset_info = {
        'dataset_name': os.path.basename(dataset_path),
        'total_samples': len(df_clean),
        'features': list(X.columns),
        'classes': classes
    }
    best_models = {
        'acc': best_acc_model,
        'f1': best_f1_model,
        'prec': best_prec_model
    }

    generate_lab5_html_report(dataset_info, all_model_stats, comparison_data, best_models, classes)

if __name__ == '__main__':
    main()
