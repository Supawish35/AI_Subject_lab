import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Mappings for categorical/string features to numerical values
CATEGORICAL_MAPPINGS = {
    "Gender": {"Male": 0, "Female": 1},
    "PreviousGrade": {"C": 0, "B": 1, "A": 2},
    "ParentalSupport": {"Low": 0, "Medium": 1, "High": 2},
}

# All feature columns to use for training/prediction
FEATURE_COLS = [
    "Gender",
    "AttendanceRate",
    "StudyHoursPerWeek",
    "PreviousGrade",
    "ExtracurricularActivities",
    "ParentalSupport",
    "Study Hours",
]

TARGET_COL = "FinalGrade"


class DataNormalizer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_cols = FEATURE_COLS
        self.target_col = TARGET_COL
        self.is_fitted = False

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert string categorical columns into numerical values."""
        df_encoded = df.copy()
        for col, mapping in CATEGORICAL_MAPPINGS.items():
            if col in df_encoded.columns:
                # Map strings to numbers; if already numeric, keep as is
                if df_encoded[col].dtype == object or df_encoded[col].dtype == "string":
                    df_encoded[col] = df_encoded[col].map(mapping)
        return df_encoded

    def fit_transform(self, df: pd.DataFrame):
        """Clean, encode strings, and normalize all features to [0, 1] while preserving class label."""
        # 1. Drop rows with missing values in required columns
        clean_df = df.dropna(subset=self.feature_cols + [self.target_col]).copy()

        # 2. Encode string features to numbers
        encoded_df = self.encode_categorical(clean_df)

        # 3. Min-Max normalize all feature columns to range [0, 1]
        encoded_df[self.feature_cols] = self.scaler.fit_transform(
            encoded_df[self.feature_cols]
        )
        self.is_fitted = True

        X = encoded_df[self.feature_cols]
        y = encoded_df[self.target_col]

        return X, y, encoded_df

    def transform_input(self, raw_input: dict | pd.DataFrame) -> pd.DataFrame:
        """Convert a single raw input sample (with strings and raw numbers) into normalized feature vector."""
        if isinstance(raw_input, dict):
            df_input = pd.DataFrame([raw_input])
        else:
            df_input = raw_input.copy()

        # Encode categorical strings
        encoded_input = self.encode_categorical(df_input)

        # Ensure all required features are present
        for col in self.feature_cols:
            if col not in encoded_input.columns:
                raise ValueError(f"Missing required feature: {col}")

        # Scale features using the fitted scaler
        encoded_input[self.feature_cols] = self.scaler.transform(
            encoded_input[self.feature_cols]
        )

        return encoded_input[self.feature_cols]


def normalize_dataset(
    input_path: str = "DataSet/student_performance_normalized.csv",
    output_path: str = "DataSet/student_performance_fully_normalized.csv",
):
    """Load dataset, encode string features to numbers, normalize all features except class, and save to CSV."""
    print(f"Reading dataset from: {input_path}")
    df = pd.read_csv(input_path)

    normalizer = DataNormalizer()
    X, y, fully_normalized_df = normalizer.fit_transform(df)

    # Save processed dataset
    fully_normalized_df.to_csv(output_path, index=False)
    print(f"Successfully normalized features and saved to: {output_path}")
    print("\nNormalized Data Sample (first 5 rows):")
    print(fully_normalized_df[FEATURE_COLS + [TARGET_COL]].head())

    return normalizer, X, y


if __name__ == "__main__":
    normalize_dataset()
