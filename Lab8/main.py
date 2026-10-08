import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from report.generator import generate_lab8_html_report
from ann_model import ANNClassifier
from reactive_form.builder import build_reactive_ann_website

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
    return pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in classes],
        columns=[f"Pred {c}" for c in classes]
    )

def evaluate_model(model_name, model_instance, X_train, y_train, X_test, y_test, classes):
    """
    Train and evaluate a single classification model on 70% Train, 30% Test split.
    Calculates Accuracy, Precision, Recall, F-measure, and Confusion Matrix.
    """
    print("\n" + "#" * 78)
    print(f" ALGORITHM: {model_name.upper()}")
    print("#" * 78)

    # Train model
    model_instance.fit(X_train, y_train)
    y_pred = model_instance.predict(X_test)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = format_cm(cm, classes)

    # 2. Performance Metrics (Weighted & Macro)
    acc = accuracy_score(y_test, y_pred)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Per-class metrics
    prec_per_class = precision_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    rec_per_class = recall_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, labels=classes, average=None, zero_division=0)

    print("\n[1] Actual Table (Confusion Matrix - Test 30%):")
    print(cm_df.to_string())

    print("\n[2] Per-Class Performance Breakdown:")
    per_class_records = []
    for idx, c in enumerate(classes):
        supp = int(np.sum(y_test == c))
        per_class_records.append({
            'Class': c,
            'Precision (%)': float(np.round(prec_per_class[idx] * 100, 2)),
            'Recall (%)': float(np.round(rec_per_class[idx] * 100, 2)),
            'F-Measure (%)': float(np.round(f1_per_class[idx] * 100, 2)),
            'Support': supp
        })
    per_class_df = pd.DataFrame(per_class_records)
    print(per_class_df.to_string(index=False))

    print("\n[3] Test Set Evaluation Summary:")
    print(f"   * Accuracy (Overall)    : {acc * 100:.2f}%")
    print(f"   * Precision (Weighted) : {prec_weighted * 100:.2f}%  |  (Macro): {prec_macro * 100:.2f}%")
    print(f"   * Recall (Weighted)    : {rec_weighted * 100:.2f}%  |  (Macro): {rec_macro * 100:.2f}%")
    print(f"   * F-Measure (Weighted) : {f1_weighted * 100:.2f}%  |  (Macro): {f1_macro * 100:.2f}%")

    loss_curve = getattr(model_instance, 'loss_curve_', None)

    return {
        'Model': model_name,
        'Accuracy': float(acc * 100),
        'Precision': float(prec_weighted * 100),
        'Recall': float(rec_weighted * 100),
        'F-Measure': float(f1_weighted * 100),
        'Precision_macro': float(prec_macro * 100),
        'Recall_macro': float(rec_macro * 100),
        'F-Measure_macro': float(f1_macro * 100),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': per_class_records,
        'model_instance': model_instance,
        'loss_curve': loss_curve
    }

def save_plots(all_model_stats, classes, base_dir):
    """Generate and save confusion matrices and comparison charts as PNG."""
    try:
        # Plot 1: Comparison Bar Chart
        models = [m['Model'] for m in all_model_stats]
        accs = [m['Accuracy'] for m in all_model_stats]
        precs = [m['Precision'] for m in all_model_stats]
        recs = [m['Recall'] for m in all_model_stats]
        f1s = [m['F-Measure'] for m in all_model_stats]

        x = np.arange(len(models))
        width = 0.2

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - 1.5 * width, accs, width, label='Accuracy', color='#4f46e5')
        ax.bar(x - 0.5 * width, precs, width, label='Precision', color='#2563eb')
        ax.bar(x + 0.5 * width, recs, width, label='Recall', color='#f59e0b')
        ax.bar(x + 1.5 * width, f1s, width, label='F-Measure', color='#10b981')

        ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
        ax.set_title('Lab 8: Model Performance Comparison (Train 70%, Test 30%)', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(min(min(accs), min(f1s)) - 10, 105)

        for i in range(len(models)):
            ax.text(x[i] - 1.5 * width, accs[i] + 1, f"{accs[i]:.1f}%", ha='center', fontsize=7, rotation=90)
            ax.text(x[i] + 1.5 * width, f1s[i] + 1, f"{f1s[i]:.1f}%", ha='center', fontsize=7, rotation=90)

        plt.tight_layout()
        chart_path = os.path.join(base_dir, 'lab8_model_comparison.png')
        plt.savefig(chart_path, dpi=200)
        plt.close()
        print(f"[✓] Saved comparison chart: {chart_path}")

        # Plot 2: 4 Confusion Matrices side-by-side
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, (m_stat, ax_sub) in enumerate(zip(all_model_stats, axes)):
            cm = np.array(m_stat['confusion_matrix'])
            im = ax_sub.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax_sub.set_title(f"{m_stat['Model']}\n(Acc: {m_stat['Accuracy']:.2f}%)", fontsize=11, fontweight='bold')
            tick_marks = np.arange(len(classes))
            ax_sub.set_xticks(tick_marks)
            ax_sub.set_xticklabels(classes)
            ax_sub.set_yticks(tick_marks)
            ax_sub.set_yticklabels(classes)
            ax_sub.set_xlabel('Predicted Label', fontweight='semibold')
            ax_sub.set_ylabel('Actual Label', fontweight='semibold')

            # Numbers in cells
            thresh = cm.max() / 2.0
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax_sub.text(c, r, format(cm[r, c], 'd'),
                                ha="center", va="center",
                                color="white" if cm[r, c] > thresh else "black",
                                fontsize=9)

        plt.tight_layout()
        cm_path = os.path.join(base_dir, 'lab8_confusion_matrices.png')
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"[✓] Saved confusion matrices image: {cm_path}")
    except Exception as e:
        print(f"[!] Warning: Plot generation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Lab 8: Artificial Neural Network (ANN) Classification & Comparison")
    parser.add_argument('--dataset', choices=['student', 'iris'], default='student',
                        help="Choose dataset: 'student' (default 10,000 students) or 'iris' (Fisher Iris)")
    parser.add_argument('--hidden', type=str, default='16,8',
                        help="ANN hidden layer sizes, e.g. '16,8' or '10' (default: 16,8)")
    parser.add_argument('--no-report', action='store_true',
                        help="Skip generating HTML report")
    parser.add_argument('--serve', action='store_true',
                        help="Start interactive reactive ANN web server after training")
    parser.add_argument('--port', type=int, default=8080,
                        help="Web server port (default: 8080)")
    args = parser.parse_args()

    print("=" * 78)
    print(" LAB 8: ARTIFICIAL NEURAL NETWORK (ANN) CLASSIFICATION & COMPARISON")
    print(" Comparing Performance: ANN vs Naive Bayes vs Decision Tree vs K-NN")
    print(" Validation Scheme: Train 70% / Test 30% Split")
    print("=" * 78)

    hidden_layers = tuple(int(x.strip()) for x in args.hidden.split(','))

    if args.dataset == 'iris':
        print("\n[1] Loading dataset: Fisher Iris Dataset (150 samples)")
        iris = load_iris()
        X_mat = iris.data
        y_vec = iris.target
        classes = [str(c) for c in iris.target_names]
        # Map 0, 1, 2 to class names
        y_vec = np.array([classes[i] for i in y_vec])
        dataset_name = "Fisher Iris Dataset"
        feature_names = iris.feature_names
    else:
        # Load unified student performance normalized dataset
        dataset_path = find_file([
            '../DataSet/student_dataset_full_normalized.csv',
            'DataSet/student_dataset_full_normalized.csv',
            '../DataSet/student_performance_normalized.csv',
            'DataSet/student_performance_normalized.csv'
        ])

        if not dataset_path or not os.path.exists(dataset_path):
            print("[!] Error: Could not locate student dataset in DataSet/ folder.")
            sys.exit(1)

        print(f"\n[1] Loading dataset: {os.path.basename(dataset_path)}")
        df = pd.read_csv(dataset_path)
        target_col = 'FinalGrade' if 'FinalGrade' in df.columns else df.columns[-1]

        drop_cols = [c for c in ['StudentID', 'Name', 'exam_score', 'placement_status'] if c in df.columns and c != target_col]
        df_clean = df.drop(columns=drop_cols).dropna()

        X = df_clean.drop(columns=[target_col]).copy()
        y = df_clean[target_col].astype(str)

        # Label encoding for categorical columns (e.g. PreviousGrade)
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
        dataset_name = os.path.basename(dataset_path)
        feature_names = list(X.columns)

    total_samples = len(y_vec)
    print(f"  * Dataset Name         : {dataset_name}")
    print(f"  * Total Samples        : {total_samples:,} rows")
    print(f"  * Features ({len(feature_names)})       : {', '.join(feature_names)}")
    print(f"  * Target Classes ({len(classes)})   : {', '.join(classes)}")
    print(f"  * Split Ratio          : Train 70% ({int(total_samples * 0.70):,} rows) / Test 30% ({int(total_samples * 0.30):,} rows)")

    # 2. Train-Test Split (Train 70%, Test 30%) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_mat, y_vec,
        test_size=0.30,
        random_state=42,
        stratify=y_vec
    )

    # Standard scaling for ANN and KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Setup Models
    # ANN Architecture: Multi-Layer Perceptron (PatternNet equivalent)
    if args.dataset == 'iris':
        ann_model = MLPClassifier(
            hidden_layer_sizes=(10,),
            activation='logistic',  # sigmoid matching Matlab patternnet
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
    else:
        ann_model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            max_iter=600,
            random_state=42,
            early_stopping=False
        )

    dt_model = DecisionTreeClassifier(random_state=42)
    nb_model = GaussianNB()
    knn_model = KNeighborsClassifier(n_neighbors=5)

    models = [
        ('ANN (Artificial Neural Network)', ann_model, True),
        ('Naive Bayes (Bayes)', nb_model, False),
        ('Decision Tree', dt_model, False),
        ('K-Nearest Neighbors (K-NN, K=5)', knn_model, True)
    ]

    all_model_stats = []

    # 4. Train and Evaluate each algorithm on 70% Train, 30% Test
    for name, model, use_scaled in models:
        xtr = X_train_scaled if use_scaled else X_train
        xte = X_test_scaled if use_scaled else X_test
        stats = evaluate_model(name, model, xtr, y_train, xte, y_test, classes)
        all_model_stats.append(stats)

    # 5. Final Comparison Table
    print("\n" + "=" * 78)
    print(" FINAL COMPARISON TABLE (TRAIN 70% / TEST 30%)")
    print("=" * 78)

    comparison_data = []
    for s in all_model_stats:
        comparison_data.append({
            'Model': s['Model'],
            'Accuracy (%)': f"{s['Accuracy']:.2f}%",
            'Precision (%)': f"{s['Precision']:.2f}%",
            'Recall (%)': f"{s['Recall']:.2f}%",
            'F-Measure (%)': f"{s['F-Measure']:.2f}%"
        })

    comp_df = pd.DataFrame(comparison_data)
    print(comp_df.to_string(index=False))
    print("=" * 78)

    # 6. Conclusions
    best_acc = max(all_model_stats, key=lambda x: x['Accuracy'])
    best_f1 = max(all_model_stats, key=lambda x: x['F-Measure'])
    best_prec = max(all_model_stats, key=lambda x: x['Precision'])

    print("\n[ Summary Conclusions ]:")
    print(f" 1. Highest Accuracy    : {best_acc['Model']} ({best_acc['Accuracy']:.2f}%)")
    print(f" 2. Highest F-Measure   : {best_f1['Model']} ({best_f1['F-Measure']:.2f}%)")
    print(f" 3. Highest Precision   : {best_prec['Model']} ({best_prec['Precision']:.2f}%)")
    print("=" * 78 + "\n")

    # 7. Generate Static Plots in report directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(base_dir, 'report')
    os.makedirs(report_dir, exist_ok=True)
    save_plots(all_model_stats, classes, report_dir)

    # 8. Generate HTML Report in report directory
    if not args.no_report:
        dataset_info = {
            'dataset_name': dataset_name,
            'total_samples': total_samples,
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'features': feature_names,
            'classes': classes
        }
        best_models = {
            'acc': best_acc,
            'f1': best_f1,
            'prec': best_prec
        }
        report_file = os.path.join(report_dir, 'lab8_report.html')
        generate_lab8_html_report(
            dataset_info,
            all_model_stats,
            comparison_data,
            best_models,
            classes,
            loss_curve=all_model_stats[0].get('loss_curve'),
            output_path=report_file
        )

    # 9. Build Light Mode Reactive Web Simulator
    try:
        reactive_dir = os.path.join(base_dir, 'reactive_form')
        os.makedirs(reactive_dir, exist_ok=True)
        index_file = os.path.join(reactive_dir, 'index.html')
        build_reactive_ann_website(index_file)
    except Exception as err:
        print(f"[!] Warning: Failed to build reactive website: {err}")

    # 10. Start Reactive Server if requested
    if args.serve:
        from reactive_form.server import start_server
        start_server(args.port)

if __name__ == '__main__':
    main()
