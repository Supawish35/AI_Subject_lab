import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Grade normalization helper
def normalize_grade(score):
    """Convert numeric score to letter grade if needed"""
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

# Configuration & Constants
DATASET_PATH = './DataSet/student_performance_normalized.csv'
FEATURES_TO_DROP = ["StudentID", "Name"]
TARGET_FEATURE = "FinalGrade"
ALL_GRADES = ["F", "D", "C", "B", "A"]

print("="*65)
print(" STUDENT PERFORMANCE DECISION TREE TRAINER")
print("="*65)

# 1. Load Dataset
print("\n[1] Loading normalized dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"  * Dataset path: {DATASET_PATH}")
print(f"  * Total rows: {len(df)}, Total columns: {len(df.columns)}")

# Ensure grade columns are letter grades
df['FinalGrade'] = df['FinalGrade'].apply(normalize_grade)
df['PreviousGrade'] = df['PreviousGrade'].apply(normalize_grade)

# Drop identifier columns
df_processed = df.copy()
if FEATURES_TO_DROP:
    print(f"  * Dropped features: {', '.join(FEATURES_TO_DROP)}")
    df_processed = df_processed.drop(columns=FEATURES_TO_DROP)

# Separate X (features) and y (target)
y = df_processed[TARGET_FEATURE]
X = df_processed.drop(columns=[TARGET_FEATURE])

# Remove rows with missing target
mask = y.notna()
X = X[mask]
y = y[mask]

# 2. Encode Categorical Features
print("\n[2] Encoding categorical features...")
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
        print(f"  * Encoded '{col}': {le.classes_.tolist()}")

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
feature_names = X.columns.tolist()

# 3. Train Decision Tree Classifier
print("\n[3] Training Decision Tree Classifier...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, y_train)

train_acc = accuracy_score(y_train, tree_classifier.predict(X_train))
test_acc = accuracy_score(y_test, tree_classifier.predict(X_test))

print(f"  * Model trained successfully!")
print(f"  * Training Accuracy: {train_acc * 100:.2f}%")
print(f"  * Testing Accuracy:  {test_acc * 100:.2f}%")

# 4. Generate & Display Decision Tree Plot
print("\n[4] Generating decision tree plot...")
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 12), dpi=100)
tree.plot_tree(
    tree_classifier,
    filled=True,
    feature_names=feature_names,
    class_names=sorted(y.unique().tolist()),
    fontsize=9
)
plt.title("Student Performance Decision Tree Classification", fontsize=14)
plt.tight_layout()
plt.show()

# 5. Output Decision Rules
print("\n" + "="*65)
print(" DECISION TREE RULES")
print("="*65)
rules = tree.export_text(tree_classifier, feature_names=feature_names)
print(rules)

print("="*65)
print(" Decision tree training complete! Use predict.py for interactive prediction.")
print("="*65)
