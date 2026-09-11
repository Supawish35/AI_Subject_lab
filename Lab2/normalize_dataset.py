import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_dataset(file_path):
    """Load dataset from CSV file."""
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully from: {file_path}")
    print(f"Original shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    return df

def normalize_grade(score):
    """Convert numeric score to letter grade."""
    if pd.isna(score):
        return np.nan
    try:
        score = float(score)
    except (ValueError, TypeError):
        return score  # Already a letter grade string

    if score < 50:
        return "F"
    elif score < 55:
        return "D"
    elif score < 65:
        return "C"
    elif score < 75:
        return "B"
    else:
        return "A"


def normalize_min_max(df, numeric_cols):
    """Normalize numeric columns to range [0, 1] using Min-Max Scaling."""
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled, scaler

def normalize_z_score(df, numeric_cols):
    """Standardize numeric columns (mean=0, std=1) using Z-Score Standardization."""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled, scaler

def process_and_normalize_dataset(
    csv_path='./DataSet/student_performance_updated_1000-selected-columns.csv',
    output_path='./DataSet/student_performance_normalized.csv',
    scaling_method='minmax'  # 'minmax' or 'zscore'
):
    # 1. Load Data
    df = load_dataset(csv_path)

    # 2. Delete rows that have null/missing values
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    deleted_rows = initial_rows - len(df)
    print(f"Deleted {deleted_rows} rows with null/missing values. Remaining rows: {len(df)}")

    # 3. Convert grade columns to Letter Grades (not scaled to numbers)
    grade_columns = [col for col in ['FinalGrade', 'PreviousGrade'] if col in df.columns]
    for col in grade_columns:
        df[col] = df[col].apply(normalize_grade)
        print(f"Normalized '{col}' to letter grades.")

    # 4. Identify numeric feature columns to scale (excluding IDs and letter grade columns)
    exclude_cols = set(grade_columns + ['StudentID', 'Name', 'Gender', 'ParentalSupport'])
    numeric_cols = [
        col for col in df.select_dtypes(include=['float64', 'int64']).columns
        if col not in exclude_cols
    ]
    print(f"\nNumeric feature columns to scale ({scaling_method}):", numeric_cols)

    # 5. Scale numerical features
    if scaling_method == 'minmax':
        print("\nApplying Min-Max Normalization (range [0, 1])...")
        df_normalized, scaler = normalize_min_max(df, numeric_cols)
    elif scaling_method == 'zscore':
        print("\nApplying Z-Score Standardization (mean=0, std=1)...")
        df_normalized, scaler = normalize_z_score(df, numeric_cols)
    else:
        raise ValueError("Invalid scaling_method. Choose 'minmax' or 'zscore'.")

    # 6. Save normalized dataset
    df_normalized.to_csv(output_path, index=False)
    print(f"\nNormalized dataset saved to: {output_path}")
    print("\nFirst 5 rows of normalized dataset:")
    print(df_normalized.head())

    return df_normalized

if __name__ == "__main__":
    process_and_normalize_dataset(
        csv_path='./DataSet/student_performance_updated_1000-selected-columns.csv',
        output_path='./DataSet/student_performance_normalized.csv',
        scaling_method='minmax'
    )
