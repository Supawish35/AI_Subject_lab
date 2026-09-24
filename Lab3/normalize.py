import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def score_to_letter_grade(score):
    """
    Convert numeric score to full 5-level letter grade (A, B, C, D, F).
    A: >= 80
    B: 70 - 79.9
    C: 60 - 69.9
    D: 50 - 59.9
    F: < 50
    """
    if pd.isna(score):
        return np.nan
    try:
        score = float(score)
    except (ValueError, TypeError):
        return str(score)

    if score >= 80.0:
        return "A"
    elif score >= 70.0:
        return "B"
    elif score >= 60.0:
        return "C"
    elif score >= 50.0:
        return "D"
    else:
        return "F"

def normalize_10k_dataset(
    raw_path=None,
    output_path=None,
    scaling_method='minmax'
):
    """
    Normalize the full 10,000-row student dataset.
    Ensures all grade levels (A, B, C, D, F) and 10,000 records are preserved.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if raw_path is None:
        raw_path = os.path.join(base_dir, '..', 'Lab1', 'archive', 'student_dataset_10000_rows.csv')
    if output_path is None:
        output_path = os.path.join(base_dir, '..', 'DataSet', 'student_dataset_full_normalized.csv')

    print("=" * 65)
    print(" 1. NORMALIZING 10,000-ROW DATASET (ALL GRADES: A, B, C, D, F)")
    print("=" * 65)
    
    if not os.path.exists(raw_path):
        print(f"[!] File not found: {raw_path}")
        return None

    df = pd.read_csv(raw_path)
    print(f"Loaded raw dataset from: {raw_path}")
    print(f"Raw shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Handle missing values if any
    df = df.dropna().reset_index(drop=True)

    # Convert continuous scores to letter grades
    df['FinalGrade'] = df['exam_score'].apply(score_to_letter_grade)
    df['PreviousGrade'] = df['previous_score'].apply(score_to_letter_grade)

    numeric_features = [
        'study_hours',
        'attendance',
        'sleep_hours',
        'internet_usage',
        'assignments_completed',
        'previous_score'
    ]
    numeric_features = [col for col in numeric_features if col in df.columns]

    # Apply scaling
    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    df_normalized = df.copy()
    df_normalized[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_normalized.to_csv(output_path, index=False)

    print(f"\nSuccessfully saved full normalized dataset to: {output_path}")
    print(f"Total rows: {len(df_normalized)}")
    print("\nGrade distribution in FinalGrade:")
    print(df_normalized['FinalGrade'].value_counts().sort_index())
    print("\nGrade distribution in PreviousGrade:")
    print(df_normalized['PreviousGrade'].value_counts().sort_index())
    print("-" * 65)

    return df_normalized

def normalize_1000_dataset(
    raw_path=None,
    output_path=None,
    scaling_method='minmax'
):
    """
    Normalize the 1,000-row student performance dataset with imputation
    to prevent losing rows and properly preserve all categories.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if raw_path is None:
        raw_path = os.path.join(base_dir, '..', 'DataSet', 'student_performance_updated_1000-selected-columns.csv')
    if output_path is None:
        output_path = os.path.join(base_dir, '..', 'DataSet', 'student_performance_normalized.csv')

    print("\n" + "=" * 65)
    print(" 2. NORMALIZING 1,000-ROW DATASET (WITH IMPUTATION)")
    print("=" * 65)

    if not os.path.exists(raw_path):
        print(f"[!] File not found: {raw_path}")
        return None

    df = pd.read_csv(raw_path)
    print(f"Loaded raw dataset from: {raw_path}")
    print(f"Raw shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Smart imputation for missing data so rows are not discarded needlessly
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype).startswith('str'):
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)
        else:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Convert numeric scores to letter grades
    df['FinalGrade'] = df['FinalGrade'].apply(score_to_letter_grade)
    df['PreviousGrade'] = df['PreviousGrade'].apply(score_to_letter_grade)

    numeric_cols = [
        'AttendanceRate',
        'StudyHoursPerWeek',
        'ExtracurricularActivities',
        'Study Hours'
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    df_normalized = df.copy()
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_normalized.to_csv(output_path, index=False)

    print(f"\nSuccessfully saved normalized dataset (all 1000 rows retained) to: {output_path}")
    print(f"Total rows: {len(df_normalized)}")
    print("\nGrade distribution in FinalGrade:")
    print(df_normalized['FinalGrade'].value_counts().sort_index())
    print("-" * 65)

    return df_normalized

if __name__ == '__main__':
    # Generate both full 10k dataset and 1k full dataset
    normalize_10k_dataset()
    normalize_1000_dataset()
