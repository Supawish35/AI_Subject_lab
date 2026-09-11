import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Dataset Paths and Configurations
RAW_DATASET_PATH = './DataSet/student_performance_updated_1000-selected-columns.csv'
NORMALIZED_DATASET_PATH = './DataSet/student_performance_normalized.csv'
FEATURES_TO_DROP = ["StudentID", "Name"]
TARGET_FEATURE = "FinalGrade"
ALL_GRADES = ["F", "D", "C", "B", "A"]

def normalize_grade(score):
    """Convert numeric score to letter grade"""
    if pd.isna(score):
        return "F"
    try:
        score = float(score)
    except (ValueError, TypeError):
        return str(score)

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

print("="*65)
print(" STUDENT PERFORMANCE GRADE PREDICTOR (DECISION TREE)")
print("="*65)

# Load normalized dataset for training
df = pd.read_csv(NORMALIZED_DATASET_PATH)

# Ensure letter grades
df['FinalGrade'] = df['FinalGrade'].apply(normalize_grade)
df['PreviousGrade'] = df['PreviousGrade'].apply(normalize_grade)

# Drop unused columns
df_processed = df.copy()
if FEATURES_TO_DROP:
    df_processed = df_processed.drop(columns=FEATURES_TO_DROP)

y = df_processed[TARGET_FEATURE]
X = df_processed.drop(columns=[TARGET_FEATURE])

# Remove rows with NaN target
mask = y.notna()
X = X[mask]
y = y[mask]

# Fit Label Encoded Categoricals
label_encoders = {}
for col in X.columns:
    col_dtype = str(X[col].dtype)
    is_categorical = X[col].dtype == 'object' or col_dtype == 'string' or col_dtype.startswith('str')
    if is_categorical:
        le = LabelEncoder()
        X[col] = X[col].fillna('missing').astype(str)
        if col == "PreviousGrade":
            all_vals = list(set(X[col].values)) + [g for g in ALL_GRADES if g not in X[col].values]
            le.fit(all_vals)
        else:
            le.fit(X[col].values)
        X[col] = le.transform(X[col].values)
        label_encoders[col] = le

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
feature_names = X.columns.tolist()

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, y_train)

train_acc = accuracy_score(y_train, tree_classifier.predict(X_train))
test_acc = accuracy_score(y_test, tree_classifier.predict(X_test))

print(f"\n[Model Information]")
print(f"  * Trained Samples: {len(X_train)}")
print(f"  * Test Accuracy:   {test_acc * 100:.2f}%\n")

# Learn Min-Max scaling boundaries from raw dataset for human-friendly input scaling
raw_df = pd.read_csv(RAW_DATASET_PATH).dropna()
raw_numeric_cols = ['AttendanceRate', 'StudyHoursPerWeek', 'ExtracurricularActivities', 'Study Hours']
raw_scalers = {}
for col in raw_numeric_cols:
    if col in raw_df.columns:
        min_val = float(raw_df[col].min())
        max_val = float(raw_df[col].max())
        raw_scalers[col] = (min_val, max_val)

def scale_human_input(col_name, val):
    """Convert human input (e.g. 85% attendance, 15 study hours) to [0, 1] scaled float."""
    if col_name in raw_scalers:
        min_v, max_v = raw_scalers[col_name]
        if max_v == min_v:
            return 0.5
        clamped_val = max(min_v, min(max_v, float(val)))
        return (clamped_val - min_v) / (max_v - min_v)
    return float(val)

# Helper functions for friendly input prompting
def prompt_choice(title, options, default_idx=1):
    """Prompt user with a numbered option menu."""
    print(f"\n[*] {title}:")
    for idx, opt in enumerate(options, 1):
        print(f"   [{idx}] {opt}")
    
    while True:
        user_in = input(f"   Select option [default: {default_idx} ({options[default_idx-1]})]: ").strip()
        if user_in == "":
            return options[default_idx-1]
        
        # Check if user typed the number
        if user_in.isdigit():
            idx = int(user_in)
            if 1 <= idx <= len(options):
                return options[idx-1]
        
        # Check if user typed option string (case-insensitive)
        for opt in options:
            if user_in.lower() == opt.lower() or user_in.lower() == opt[0].lower():
                return opt
        
        print(f"   [!] Invalid input '{user_in}'. Please choose a number (1-{len(options)}) or name.")

def prompt_number(title, hint, default, min_val=None, max_val=None):
    """Prompt user for numerical input with validation."""
    print(f"\n[*] {title}:")
    print(f"   {hint}")
    while True:
        user_in = input(f"   Enter value [default: {default}]: ").strip()
        if user_in == "":
            return float(default)
        
        user_in_clean = user_in.replace("%", "").strip()
        try:
            val = float(user_in_clean)
            if min_val is not None and val < min_val:
                print(f"   [!] Value must be at least {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"   [!] Value cannot exceed {max_val}.")
                continue
            return val
        except ValueError:
            print(f"   [!] Invalid number '{user_in}'. Please enter a valid number.")

# Interactive User Input Section
print("="*65)
print(" ENTER STUDENT INFORMATION")
print("="*65)

human_inputs = {}

# 1. Gender
human_inputs['Gender'] = prompt_choice("Gender", ["Male", "Female"], default_idx=1)

# 2. Attendance Rate
human_inputs['AttendanceRate'] = prompt_number(
    "Attendance Rate",
    "Enter class attendance percentage (e.g. 85 for 85%) [Range: 70% - 100%]",
    default=85.0,
    min_val=50.0,
    max_val=100.0
)

# 3. Weekly Study Hours
human_inputs['StudyHoursPerWeek'] = prompt_number(
    "Weekly Study Hours",
    "Enter total hours spent studying per week (e.g. 18 hours) [Range: 8 - 30 hrs]",
    default=18.0,
    min_val=0.0,
    max_val=100.0
)

# 4. Previous Grade
human_inputs['PreviousGrade'] = prompt_choice(
    "Previous Academic Grade",
    ["A (85+)", "B (75-84)", "C (65-74)", "D (50-64)", "F (<50)"],
    default_idx=2
).split()[0]  # Extract grade letter 'A', 'B', 'C', etc.

# 5. Extracurricular Activities
human_inputs['ExtracurricularActivities'] = prompt_number(
    "Extracurricular Activities",
    "Enter number of extracurricular activities joined (0 to 3)",
    default=1.0,
    min_val=0.0,
    max_val=10.0
)

# 6. Parental Support
human_inputs['ParentalSupport'] = prompt_choice(
    "Parental Support Level",
    ["High", "Medium", "Low"],
    default_idx=2
)

# 7. Daily Study Hours
human_inputs['Study Hours'] = prompt_number(
    "Daily Study Hours",
    "Enter average study hours per day (e.g. 2.5 hours) [Range: 0 - 5 hrs]",
    default=2.5,
    min_val=0.0,
    max_val=24.0
)

# Convert Human Inputs to Model Input DataFrame
model_inputs = {}
for col in feature_names:
    raw_v = human_inputs[col]
    if col in raw_numeric_cols:
        model_inputs[col] = scale_human_input(col, raw_v)
    else:
        model_inputs[col] = raw_v

# Create DataFrame
sample_df = pd.DataFrame([model_inputs])

# Encode Categorical Features
for col, le in label_encoders.items():
    if col in sample_df.columns:
        val = str(sample_df[col].iloc[0])
        if val in le.classes_:
            sample_df[col] = le.transform([val])
        else:
            sample_df[col] = 0

sample_df = sample_df.apply(pd.to_numeric, errors='coerce').fillna(0)
sample_df = sample_df.reindex(columns=feature_names, fill_value=0)

# Make Prediction
predicted_grade = tree_classifier.predict(sample_df)[0]
probabilities = tree_classifier.predict_proba(sample_df)[0]

print("\n" + "="*65)
print(" PREDICTION RESULT SUMMARY")
print("="*65)
print(f" PREDICTED FINAL GRADE : [ {predicted_grade} ]")
print("="*65)

print("\nStudent Profile Provided:")
print(f"   - Gender:                   {human_inputs['Gender']}")
print(f"   - Attendance Rate:          {human_inputs['AttendanceRate']:.1f}%")
print(f"   - Weekly Study Hours:       {human_inputs['StudyHoursPerWeek']:.1f} hrs")
print(f"   - Previous Grade:           {human_inputs['PreviousGrade']}")
print(f"   - Extracurriculars:         {int(human_inputs['ExtracurricularActivities'])} activities")
print(f"   - Parental Support:         {human_inputs['ParentalSupport']}")
print(f"   - Daily Study Hours:        {human_inputs['Study Hours']:.1f} hrs")

print("\nInternal Model Inputs (Scaled [0, 1]):")
for col in feature_names:
    val = model_inputs[col]
    if isinstance(val, float):
        print(f"   - {col:27s}: {val:.4f}")
    else:
        print(f"   - {col:27s}: {val}")

print("\n" + "="*65)
