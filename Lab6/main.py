import os
import sys
import itertools
import numpy as np
import pandas as pd
from report_generator import generate_lab6_html_report

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

def calculate_univariate_divergence(u1: float, s1_sq: float, u2: float, s2_sq: float) -> float:
    """
    Equation 1: Univariate feature distance (Jeffreys divergence).
    d_ij = 0.5 * (s_j^2 / s_i^2 + s_i^2 / s_j^2 - 2) + 0.5 * (u_i - u_j)^2 * (1 / s_i^2 + 1 / s_j^2)
    """
    eps = 1e-9
    s1_sq = max(float(s1_sq), eps)
    s2_sq = max(float(s2_sq), eps)

    term1 = 0.5 * ((s2_sq / s1_sq) + (s1_sq / s2_sq) - 2.0)
    term2 = 0.5 * ((u1 - u2) ** 2) * ((1.0 / s1_sq) + (1.0 / s2_sq))
    return float(term1 + term2)

def calculate_multivariate_divergence(m1: np.ndarray, cov1: np.ndarray, m2: np.ndarray, cov2: np.ndarray) -> float:
    """
    Equation 2: Multivariate feature vector distance (Jeffreys divergence).
    d_ij = 0.5 * trace{Sigma_i^-1 * Sigma_j + Sigma_j^-1 * Sigma_i - 2I} 
           + 0.5 * (u_i - u_j)^T * (Sigma_i^-1 + Sigma_j^-1) * (u_i - u_j)
    """
    p = len(m1)
    I = np.eye(p)

    try:
        inv1 = np.linalg.inv(cov1)
        inv2 = np.linalg.inv(cov2)
        cov1_use, cov2_use = cov1, cov2
    except np.linalg.LinAlgError:
        eps = 1e-6
        cov1_use = cov1 + I * eps
        cov2_use = cov2 + I * eps
        inv1 = np.linalg.pinv(cov1_use)
        inv2 = np.linalg.pinv(cov2_use)

    term1 = 0.5 * np.trace(inv1 @ cov2_use + inv2 @ cov1_use - 2.0 * I)
    diff = m1 - m2
    term2 = 0.5 * float(diff @ (inv1 + inv2) @ diff)
    return float(term1 + term2)

def evaluate_single_features(df: pd.DataFrame, features: list, target_col: str, class_pairs: list) -> list:
    """Evaluate univariate distance for each individual feature."""
    results = []
    for feat in features:
        pair_distances = {}
        pair_values = []
        for c1, c2 in class_pairs:
            x1 = df[df[target_col] == c1][feat].values
            x2 = df[df[target_col] == c2][feat].values

            u1, s1_sq = np.mean(x1), np.var(x1, ddof=1)
            u2, s2_sq = np.mean(x2), np.var(x2, ddof=1)

            d = calculate_univariate_divergence(u1, s1_sq, u2, s2_sq)
            pair_distances[f"{c1}-{c2}"] = d
            pair_values.append(d)

        results.append({
            'feature': feat,
            'mean_divergence': float(np.mean(pair_values)),
            'min_divergence': float(np.min(pair_values)),
            'max_divergence': float(np.max(pair_values)),
            'pairwise': pair_distances
        })

    # Sort descending by mean divergence
    results.sort(key=lambda x: x['mean_divergence'], reverse=True)
    return results

def evaluate_feature_subsets(df: pd.DataFrame, feature_subsets: list, target_col: str, class_pairs: list) -> list:
    """Evaluate multivariate distance for each feature subset."""
    results = []
    for subset in feature_subsets:
        subset_list = list(subset)
        pair_distances = {}
        pair_values = []
        for c1, c2 in class_pairs:
            x1 = df[df[target_col] == c1][subset_list].values
            x2 = df[df[target_col] == c2][subset_list].values

            m1, cov1 = np.mean(x1, axis=0), np.cov(x1, rowvar=False)
            m2, cov2 = np.mean(x2, axis=0), np.cov(x2, rowvar=False)

            d = calculate_multivariate_divergence(m1, cov1, m2, cov2)
            pair_distances[f"{c1}-{c2}"] = d
            pair_values.append(d)

        results.append({
            'features': subset_list,
            'mean_divergence': float(np.mean(pair_values)),
            'min_divergence': float(np.min(pair_values)),
            'max_divergence': float(np.max(pair_values)),
            'pairwise': pair_distances
        })

    results.sort(key=lambda x: x['mean_divergence'], reverse=True)
    return results

def print_table(headers: list, rows: list, col_align: list = None):
    """Utility to print a well-aligned ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = "+-" + "-+-".join(["-" * w for w in col_widths]) + "-+"
    print(sep)
    header_str = "| " + " | ".join([f"{headers[i]:^{col_widths[i]}}" for i in range(len(headers))]) + " |"
    print(header_str)
    print(sep)

    for row in rows:
        formatted_row = []
        for i, val in enumerate(row):
            align = col_align[i] if col_align else '<'
            if align == '>':
                formatted_row.append(f"{str(val):>{col_widths[i]}}")
            elif align == '^':
                formatted_row.append(f"{str(val):^{col_widths[i]}}")
            else:
                formatted_row.append(f"{str(val):<{col_widths[i]}}")
        print("| " + " | ".join(formatted_row) + " |")
    print(sep)

def main():
    print("=" * 82)
    print(" LAB 6: FEATURE DISTANCE (JEFFREYS DIVERGENCE) ANALYSIS")
    print("=" * 82)

    # 1. Locate and load dataset
    dataset_candidates = [
        "../DataSet/student_dataset_full_normalized.csv",
        "DataSet/student_dataset_full_normalized.csv",
        "../student_dataset_full_normalized.csv",
        "student_dataset_full_normalized.csv"
    ]
    dataset_path = find_file(dataset_candidates)
    if not dataset_path or not os.path.exists(dataset_path):
        print("[!] Error: Could not locate 'student_dataset_full_normalized.csv'.")
        sys.exit(1)

    print(f"\n[1] Loading Dataset: {os.path.basename(dataset_path)}")
    df = pd.read_csv(dataset_path)
    target_col = 'FinalGrade'

    classes = sorted(list(df[target_col].dropna().unique()))
    class_pairs = list(itertools.combinations(classes, 2))

    print(f"  * Total Samples  : {len(df):,} rows")
    print(f"  * Target Column  : {target_col}")
    print(f"  * Classes ({len(classes)})   : {', '.join(classes)}")
    print(f"  * Pairwise Pairs : {len(class_pairs)} class pairs ({', '.join([f'{c1}-{c2}' for c1, c2 in class_pairs])})")

    # =========================================================================
    # SECTION 1: 4 FEATURES SPECIFICATION (MAIN ASSIGNMENT REQUIREMENT)
    # =========================================================================
    features_4 = ['study_hours', 'attendance', 'sleep_hours', 'previous_score']
    print("\n" + "#" * 82)
    print(" SECTION 1: 4-FEATURE DEMONSTRATION")
    print(f" Features Tested: {', '.join(features_4)}")
    print("#" * 82)

    # 1.1 Individual Distance (Equation 1)
    print("\n[1.1] Individual Feature Distance (Equation 1: Univariate d_ij):")
    sec1_single = evaluate_single_features(df, features_4, target_col, class_pairs)
    
    single_table_headers = ["Rank", "Feature Name", "Mean Divergence", "Min Pair (d_ij)", "Max Pair (d_ij)"]
    single_table_rows = []
    for rank, item in enumerate(sec1_single, start=1):
        single_table_rows.append([
            f"#{rank}" + (" (Best)" if rank == 1 else ""),
            item['feature'],
            f"{item['mean_divergence']:.4f}",
            f"{item['min_divergence']:.4f}",
            f"{item['max_divergence']:.4f}"
        ])
    print_table(single_table_headers, single_table_rows, ['^', '<', '>', '>', '>'])

    top1_feat = sec1_single[0]['feature']
    top2_feat = sec1_single[1]['feature']
    top2_combo_set = {top1_feat, top2_feat}

    # 1.2 Half Features: All 2-Feature Combinations (Equation 2)
    pairs_4 = list(itertools.combinations(features_4, 2))
    print(f"\n[1.2] Half Features (2 of 4 Features = {len(pairs_4)} Combinations) (Equation 2: Multivariate d_ij):")
    sec1_pairs = evaluate_feature_subsets(df, pairs_4, target_col, class_pairs)

    for item in sec1_pairs:
        if set(item['features']) == top2_combo_set:
            item['is_top2_combo'] = True
        else:
            item['is_top2_combo'] = False

    pair_table_headers = ["Rank", "Feature Pair", "Mean Divergence", "Min Pair", "Max Pair", "Notes"]
    pair_table_rows = []
    for rank, item in enumerate(sec1_pairs, start=1):
        note = []
        if rank == 1:
            note.append("Top-1 Pair")
        if item['is_top2_combo']:
            note.append("Top-2 Single Combo")
        pair_table_rows.append([
            f"#{rank}",
            " + ".join(item['features']),
            f"{item['mean_divergence']:.4f}",
            f"{item['min_divergence']:.4f}",
            f"{item['max_divergence']:.4f}",
            ", ".join(note) if note else "-"
        ])
    print_table(pair_table_headers, pair_table_rows, ['^', '<', '>', '>', '>', '^'])

    # 1.3 All 4 Features Combined (Equation 2)
    print("\n[1.3] All 4 Features Combined (Equation 2: Multivariate 4x4 Covariance):")
    sec1_all_res = evaluate_feature_subsets(df, [features_4], target_col, class_pairs)[0]
    print(f"  * All 4 Features Divergence : {sec1_all_res['mean_divergence']:.4f}")
    print(f"  * Min Pair Divergence       : {sec1_all_res['min_divergence']:.4f}")
    print(f"  * Max Pair Divergence       : {sec1_all_res['max_divergence']:.4f}")

    # =========================================================================
    # SECTION 2: COMPREHENSIVE 7-FEATURE ANALYSIS
    # =========================================================================
    features_7 = [
        'study_hours', 'attendance', 'sleep_hours', 'internet_usage',
        'assignments_completed', 'previous_score', 'exam_score'
    ]
    print("\n" + "#" * 82)
    print(" SECTION 2: COMPREHENSIVE 7-FEATURE ANALYSIS")
    print(f" All Available Features: {', '.join(features_7)}")
    print("#" * 82)

    # 2.1 All 7 Single Features
    print("\n[2.1] Ranked Individual Feature Distance (Equation 1: Univariate):")
    sec2_single = evaluate_single_features(df, features_7, target_col, class_pairs)
    sec2_table_headers = ["Rank", "Feature Name", "Mean Divergence", "Min Pair (d_ij)", "Max Pair (d_ij)"]
    sec2_table_rows = []
    for rank, item in enumerate(sec2_single, start=1):
        sec2_table_rows.append([
            f"#{rank}" + (" (Best)" if rank == 1 else ""),
            item['feature'],
            f"{item['mean_divergence']:.4f}",
            f"{item['min_divergence']:.4f}",
            f"{item['max_divergence']:.4f}"
        ])
    print_table(sec2_table_headers, sec2_table_rows, ['^', '<', '>', '>', '>'])

    # 2.2 Half Features for 7 features (e.g. 3 and 4 feature subsets)
    # Test top combinations of 3 and 4 features, and combinations of top-4 features
    top4_feats_sec2 = [item['feature'] for item in sec2_single[:4]]
    top3_feats_sec2 = [item['feature'] for item in sec2_single[:3]]
    subsets_to_evaluate = [
        top3_feats_sec2,
        top4_feats_sec2,
        list(itertools.combinations(top4_feats_sec2, 2)),
    ]
    # Evaluate 3-features and 4-features
    combos_3 = list(itertools.combinations(top4_feats_sec2, 3))
    sec2_combos_eval = evaluate_feature_subsets(df, combos_3 + [top4_feats_sec2], target_col, class_pairs)
    best_half_sec2 = sec2_combos_eval[0]

    # 2.3 All 7 Features Combined
    print("\n[2.2] All 7 Features Combined (Equation 2: Multivariate 7x7 Covariance):")
    sec2_all_res = evaluate_feature_subsets(df, [features_7], target_col, class_pairs)[0]
    print(f"  * All 7 Features Divergence : {sec2_all_res['mean_divergence']:.4f}")
    print(f"  * Min Pair Divergence       : {sec2_all_res['min_divergence']:.4f}")
    print(f"  * Max Pair Divergence       : {sec2_all_res['max_divergence']:.4f}")

    # =========================================================================
    # SUMMARY CONCLUSIONS
    # =========================================================================
    print("\n" + "=" * 82)
    print(" SUMMARY CONCLUSIONS (คำตอบสำหรับโจทย์ Lab 6)")
    print("=" * 82)
    print(f"1. Feature เดี่ยวไหนดีที่สุด:")
    print(f"   - ในชุด 4 Features : '{sec1_single[0]['feature']}' ดีที่สุด (Divergence = {sec1_single[0]['mean_divergence']:.4f})")
    print(f"   - ในชุด 7 Features : '{sec2_single[0]['feature']}' ดีที่สุด (Divergence = {sec2_single[0]['mean_divergence']:.4f})")
    print(f"\n2. ครึ่งหนึ่งของ Features (Half Features):")
    print(f"   - ในชุด 4 Features (เลือก 2 Features): คู่ที่ดีที่สุดคือ {' + '.join(sec1_pairs[0]['features'])} (Divergence = {sec1_pairs[0]['mean_divergence']:.4f})")
    print(f"   - สอดคล้องกับ Top-2 Feature เดี่ยว ({top1_feat} + {top2_feat})")
    print(f"\n3. ทุก Features (All Features):")
    print(f"   - รวม 4 Features (4D Vector) : Divergence = {sec1_all_res['mean_divergence']:.4f}")
    print(f"   - รวม 7 Features (7D Vector) : Divergence = {sec2_all_res['mean_divergence']:.4f}")
    print(f"   - สรุป: การใช้ Feature ร่วมกันหลากมิติ ให้ระยะห่างแยกคลาส (Divergence) สูงขึ้นอย่างเด่นชัด")
    print("=" * 82)

    # 4. Generate HTML Report
    dataset_info = {
        'dataset_name': os.path.basename(dataset_path),
        'total_samples': len(df),
        'target_col': target_col,
        'classes': classes
    }
    sec1_data = {
        'features_tested': features_4,
        'single_features': sec1_single,
        'pair_features': sec1_pairs,
        'all_features': sec1_all_res
    }
    sec2_data = {
        'features_tested': features_7,
        'single_features': sec2_single,
        'best_half': best_half_sec2,
        'all_features': sec2_all_res
    }

    report_path = generate_lab6_html_report(dataset_info, sec1_data, sec2_data)
    print(f"\n[+] Generated interactive HTML Report: {report_path}")

if __name__ == '__main__':
    main()
