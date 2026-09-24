import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def find_dataset_path(preferred_filename='student_dataset_full_normalized.csv'):
    """Find dataset file in possible project locations, with fallback options."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, '..', 'DataSet', preferred_filename),
        os.path.join(current_dir, 'DataSet', preferred_filename),
        os.path.join(os.getcwd(), 'DataSet', preferred_filename),
        os.path.join(os.getcwd(), '..', 'DataSet', preferred_filename),
        os.path.join(current_dir, '..', 'DataSet', 'student_performance_normalized.csv'),
        os.path.join(current_dir, 'DataSet', 'student_performance_normalized.csv'),
        preferred_filename
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None

def main():
    print("=" * 65)
    print(" LAB 3: NAIVE BAYES CLASSIFICATION (การจำแนกข้อมูลด้วยเบย์)")
    print("=" * 65)

    # 1. โหลดชุดข้อมูล (Load Full Normalized Dataset with all grades A, B, C, D, F)
    preferred_file = 'student_dataset_full_normalized.csv'
    file_path = find_dataset_path(preferred_file)

    if file_path is None:
        print(f"[!] Warning: ไม่พบ '{preferred_file}' กำลังลองสร้างไฟล์จาก normalize.py...")
        try:
            from normalize import normalize_10k_dataset
            normalize_10k_dataset()
            file_path = find_dataset_path(preferred_file)
        except Exception as e:
            print(f"[!] Error: {e}")

    if file_path is None:
        print(f"[!] Error: ไม่สามารถค้นหาหรือสร้างไฟล์ Dataset ได้")
        sys.exit(1)

    print(f"\n[1] โหลดชุดข้อมูลจาก: {file_path}")
    df = pd.read_csv(file_path)
    print(f"  * จำนวนแถวทั้งหมด (Total Rows)   : {df.shape[0]:,}")
    print(f"  * จำนวนคอลัมน์ (Total Columns) : {df.shape[1]}")
    print("\n--- ตัวอย่างข้อมูล 5 แถวแรก ---")
    print(df.head())

    # 2. เตรียม Features และ Target
    print("\n[2] เตรียมคุณลักษณะ (Features) และตัวแปรเป้าหมาย (Target)...")
    target_column = 'FinalGrade' if 'FinalGrade' in df.columns else df.columns[-1]

    # คัดเลือกคอลัมน์ที่ไม่จำเป็นออก (IDs, continuous target duplicates, placement labels if not features)
    drop_candidates = ['StudentID', 'Name', 'exam_score', 'placement_status']
    drop_cols = [col for col in drop_candidates if col in df.columns and col != target_column]
    df_cleaned = df.drop(columns=drop_cols).dropna().copy()
    
    if drop_cols:
        print(f"  * นำคอลัมน์ที่ไม่จำเป็นออก: {drop_cols}")
    print(f"  * จำนวนข้อมูลหลังจัดการ Missing Values: {len(df_cleaned):,} แถว")

    X = df_cleaned.drop(columns=[target_column]).copy()
    y = df_cleaned[target_column].astype(str)

    # แปลง Categorical Features เป็นตัวเลขด้วย LabelEncoder
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype).startswith('str'):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            print(f"  * แปลงคอลัมน์ '{col}' ด้วย LabelEncoder: {list(le.classes_)}")

    X = X.astype(float)
    feature_names = list(X.columns)

    print(f"\n  * Features ({len(feature_names)} ตัว): {', '.join(feature_names)}")
    print(f"  * Target Class: '{target_column}' -> คลาสเกรดทั้งหมด: {sorted(y.unique())}")
    print("\n  * สัดส่วนคลาสเป้าหมายในชุดข้อมูล:")
    print(y.value_counts().sort_index().to_string())

    # 3. แบ่งชุดข้อมูลเป็น Train และ Test Set (80:20)
    print("\n[3] แบ่งข้อมูลเป็น Training (80%) และ Testing (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  * Training Samples: {len(X_train):,} แถว")
    print(f"  * Testing Samples : {len(X_test):,} แถว")

    # 4. สร้างและเทรนโมเดล Naive Bayes (Gaussian Naive Bayes)
    print("\n[4] สร้างและฝึกสอนโมเดล Gaussian Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    print(f"  * ความน่าจะเป็นก่อนหน้าของแต่ละคลาส (Prior Probabilities P(C)):")
    for cls_name, prior in zip(nb_model.classes_, nb_model.class_prior_):
        print(f"      - Grade '{cls_name}': {prior:.4f} ({prior*100:.2f}%)")

    # 5. ทำนายผลบนชุดข้อมูลทดสอบ (Predict)
    print("\n[5] ทำนายผลบนชุดข้อมูลทดสอบ (Testing Data)...")
    y_pred = nb_model.predict(X_test)
    y_pred_proba = nb_model.predict_proba(X_test)

    # 6. ประเมินประสิทธิภาพของโมเดล (Model Evaluation)
    print("\n" + "=" * 65)
    print(" ผลการประเมินประสิทธิภาพโมเดล (MODEL EVALUATION)")
    print("=" * 65)

    train_acc = accuracy_score(y_train, nb_model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    print(f" Training Accuracy : {train_acc * 100:.2f}%")
    print(f" Testing Accuracy  : {test_acc * 100:.2f}%")
    print("=" * 65)

    print("\n--- รายงานการจำแนกประเภท (Classification Report) ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("--- เมทริกซ์ความสับสน (Confusion Matrix) ---")
    cm = confusion_matrix(y_test, y_pred, labels=nb_model.classes_)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in nb_model.classes_],
        columns=[f"Pred {c}" for c in nb_model.classes_]
    )
    print(cm_df)

    # 7. เปรียบเทียบผลการทำนาย 10 แถวแรก
    print("\n--- ตัวอย่างผลการทำนาย 10 แถวแรก (Predicted vs Actual) ---")
    sample_compare = pd.DataFrame({
        'Actual': y_test.iloc[:10].values,
        'Predicted': y_pred[:10],
        'Result': ['✓' if a == p else '✗' for a, p in zip(y_test.iloc[:10].values, y_pred[:10])]
    })
    for idx, cls_name in enumerate(nb_model.classes_):
        sample_compare[f'P({cls_name})'] = np.round(y_pred_proba[:10, idx], 3)
    print(sample_compare.to_string(index=False))

    # 8. ตัวอย่างการทำนายผลข้อมูลใหม่ (Sample Prediction)
    print("\n" + "=" * 65)
    print(" ตัวอย่างการทำนายผลข้อมูลใหม่ (CUSTOM SAMPLE PREDICTION)")
    print("=" * 65)

    # สร้างตัวอย่าง Input สำหรับทดสอบ
    sample_input = {}
    for feat in feature_names:
        sample_input[feat] = float(X_test.iloc[0][feat])
    
    sample_df = pd.DataFrame([sample_input])[feature_names]
    pred_grade = nb_model.predict(sample_df)[0]
    pred_probs = nb_model.predict_proba(sample_df)[0]

    print("ข้อมูล Feature ที่นำมาทดสอบ:")
    for feat, val in sample_input.items():
        print(f"  - {feat:25s}: {val:.4f}")

    print(f"\n>> ผลการจำแนกเกรด (Predicted Grade) : [ {pred_grade} ]")
    print(">> ความน่าจะเป็นของแต่ละเกรด (Posterior Probabilities):")
    for cls_name, prob in zip(nb_model.classes_, pred_probs):
        print(f"     P(FinalGrade = '{cls_name}' | Features) = {prob*100:.2f}%")
    print("=" * 65)

if __name__ == '__main__':
    main()
