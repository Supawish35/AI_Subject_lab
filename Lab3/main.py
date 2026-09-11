import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. โหลดชุดข้อมูล (Load Normalized Dataset)
file_path = os.path.join(os.path.dirname(__file__), 'DataSet', 'student_performance_normalized.csv')
if not os.path.exists(file_path):
    file_path = 'DataSet/student_performance_normalized.csv'

df = pd.read_csv(file_path)

# จัดการข้อมูลที่สูญหาย (ถ้ามี)
df = df.dropna()

print("--- ข้อมูลตัวอย่าง 5 แถวแรก ---")
print(df.head())
print()

# 2. เตรียมคุณลักษณะ (Features) และตัวแปรเป้าหมาย (Target)
# ลบคอลัมน์ที่ไม่จำเป็นสำหรับการฝึกสอน เช่น StudentID และ Name
drop_cols = [col for col in ['StudentID', 'Name'] if col in df.columns]
df_cleaned = df.drop(columns=drop_cols)

# แยก Features (X) และ Target (y) โดยให้ 'FinalGrade' เป็นคลาสเป้าหมาย
target_column = 'FinalGrade'
X = df_cleaned.drop(columns=[target_column])
y = df_cleaned[target_column]

# แปลงคอลัมน์ที่เป็น Categorical (ข้อความ) ให้เป็นตัวเลขด้วย LabelEncoder
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# 3. แบ่งชุดข้อมูลเป็น Training Data และ Test Data (เช่น 80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. สร้างและฝึกโมเดล Naive Bayes (Gaussian Naive Bayes)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 5. ให้โมเดลทำนายค่า labels ของข้อมูลทดสอบ (Predict)
predicted_labels = nb_model.predict(X_test)

# 6. แสดงผลลัพธ์การทำนายและการประเมินประสิทธิภาพ
print("--- ผลลัพธ์การทำนาย (Predicted vs Actual) 10 แถวแรก ---")
comparison_df = pd.DataFrame({
    'Actual': y_test.iloc[:10].values,
    'Predicted': predicted_labels[:10]
})
print(comparison_df)
print()

# คำนวณค่าความแม่นยำ (Accuracy)
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print()

print("--- Classification Report ---")
print(classification_report(y_test, predicted_labels))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, predicted_labels))
