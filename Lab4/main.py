import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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

class KNNClassifier:
    """
    K-Nearest Neighbors (K-NN) Classifier implemented with NumPy
    คำนวณระยะห่างแบบ Euclidean Distance และโหวตคลาสข้างเคียง
    """
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)

    def _predict_single(self, test_vector: np.ndarray):
        """
        คำนวณ Euclidean distance กับข้อมูลทั้งหมดใน Training set
        แล้วหา K เพื่อนบ้านที่ใกล้ที่สุดเพื่อลงคะแนนเสียง (Majority Vote)
        """
        # 1. คำนวณระยะห่าง Euclidean Distance: sqrt(sum((x - y)^2))
        distances = np.sqrt(np.sum((self.X_train - test_vector) ** 2, axis=1))

        # 2. เรียงลำดับดัชนีของระยะห่างจากน้อยไปมาก
        sorted_indices = np.argsort(distances)

        # 3. เลือก K เพื่อนบ้านที่ใกล้ที่สุด
        nearest_k_indices = sorted_indices[:self.k]
        nearest_k_distances = distances[nearest_k_indices]
        nearest_k_classes = self.y_train[nearest_k_indices]

        # 4. นับจำนวนและหาคลาสที่มีเพื่อนบ้านมากที่สุด (Majority Vote)
        vote_counts = Counter(nearest_k_classes)
        predicted_class = vote_counts.most_common(1)[0][0]

        return predicted_class, nearest_k_indices, nearest_k_distances, nearest_k_classes, vote_counts

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test = np.array(X_test, dtype=float)
        predictions = [self._predict_single(row)[0] for row in X_test]
        return np.array(predictions)

    def predict_detailed(self, test_vector: np.ndarray):
        return self._predict_single(np.array(test_vector, dtype=float))

class Lab4KNNApp:
    def __init__(self, default_k: int = 5):
        self.k = default_k
        self.knn = KNNClassifier(k=self.k)
        self.feature_names = []
        self.raw_scalers = {}
        self.label_encoders = {}
        self.classes = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_train(self):
        dataset_path = find_file([
            '../DataSet/student_dataset_full_normalized.csv',
            'DataSet/student_dataset_full_normalized.csv',
            '../DataSet/student_performance_normalized.csv',
            'DataSet/student_performance_normalized.csv'
        ])

        if not dataset_path:
            print("[*] Dataset not found, trying to generate via Lab3 normalize.py...")
            try:
                from normalize import normalize_10k_dataset
                normalize_10k_dataset()
                dataset_path = find_file(['../DataSet/student_dataset_full_normalized.csv', 'DataSet/student_dataset_full_normalized.csv'])
            except Exception:
                pass

        if not dataset_path or not os.path.exists(dataset_path):
            print("[!] Error: ไม่พบไฟล์ Dataset ในโฟลเดอร์ DataSet")
            sys.exit(1)

        print("=" * 68)
        print(" LAB 4: K-NEAREST NEIGHBORS (K-NN) CLASSIFICATION")
        print(" การจำแนกข้อมูลด้วยระเบียบวิธีเพื่อนบ้านที่ใกล้ที่สุด")
        print("=" * 68)
        print(f"[*] โหลดชุดข้อมูลจาก: {os.path.basename(dataset_path)}")

        df = pd.read_csv(dataset_path)
        target_col = 'FinalGrade' if 'FinalGrade' in df.columns else df.columns[-1]

        drop_cols = [c for c in ['StudentID', 'Name', 'exam_score', 'placement_status'] if c in df.columns and c != target_col]
        df_clean = df.drop(columns=drop_cols).dropna()

        X = df_clean.drop(columns=[target_col]).copy()
        y = df_clean[target_col].astype(str)

        for col in X.columns:
            if X[col].dtype == 'object' or str(X[col].dtype).startswith('str'):
                le = LabelEncoder()
                if col == 'PreviousGrade':
                    le.fit(['A', 'B', 'C', 'D', 'F'])
                    X[col] = le.transform(X[col].astype(str))
                else:
                    X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le

        X = X.astype(float)
        self.feature_names = list(X.columns)
        self.classes = sorted(list(y.unique()))

        # Train / Test split (80:20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X.values, y.values, test_size=0.2, random_state=42, stratify=y.values
        )

        self.knn.fit(self.X_train, self.y_train)

        # Learn raw bounds for human input scaling
        raw_path = find_file([
            '../Lab1/archive/student_dataset_10000_rows.csv',
            'Lab1/archive/student_dataset_10000_rows.csv',
            '../DataSet/student_performance_updated_1000-selected-columns.csv'
        ])
        if raw_path:
            raw_df = pd.read_csv(raw_path)
            for col in self.feature_names:
                if col in raw_df.columns and np.issubdtype(raw_df[col].dtype, np.number):
                    self.raw_scalers[col] = (float(raw_df[col].min()), float(raw_df[col].max()))

        default_bounds = {
            'study_hours': (1.0, 11.0),
            'attendance': (40.0, 100.0),
            'sleep_hours': (3.0, 10.0),
            'internet_usage': (1.0, 8.0),
            'assignments_completed': (0.0, 10.0),
            'previous_score': (35.0, 95.0),
            'AttendanceRate': (50.0, 100.0),
            'StudyHoursPerWeek': (5.0, 40.0),
            'ExtracurricularActivities': (0.0, 5.0),
            'Study Hours': (0.0, 5.0)
        }
        for k_b, v_b in default_bounds.items():
            if k_b not in self.raw_scalers:
                self.raw_scalers[k_b] = v_b

        print(f"[*] เตรียมโมเดล K-NN เรียบร้อย (ค่าเริ่มต้น K = {self.k})")
        print(f"    - Training Samples : {len(self.X_train):,} ตัวอย่าง")
        print(f"    - Testing Samples  : {len(self.X_test):,} ตัวอย่าง")
        print(f"    - Features ({len(self.feature_names)} ตัว): {', '.join(self.feature_names)}")
        print(f"    - Target Classes   : {', '.join(self.classes)}")
        print("=" * 68)

    def evaluate_model(self):
        print("\n" + "=" * 68)
        print(f" ประเมินประสิทธิภาพโมเดล K-NN (K = {self.k})")
        print("=" * 68)
        y_pred = self.knn.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print(f" Model Test Accuracy: {acc * 100:.2f}%\n")
        print("--- Classification Report ---")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        print("--- Confusion Matrix ---")
        cm = confusion_matrix(self.y_test, y_pred, labels=self.classes)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {c}" for c in self.classes],
            columns=[f"Pred {c}" for c in self.classes]
        )
        print(cm_df)
        print("=" * 68)

    def scale_feature(self, name, val):
        if name in self.raw_scalers:
            min_v, max_v = self.raw_scalers[name]
            if max_v == min_v:
                return 0.5
            clamped = max(min_v, min(max_v, float(val)))
            return (clamped - min_v) / (max_v - min_v)
        return float(val)

    def prompt_number(self, title, hint, default, min_val=None, max_val=None):
        print(f"\n[*] {title}:")
        print(f"   {hint}")
        while True:
            u_in = input(f"   ระบุค่า [default: {default}]: ").strip()
            if u_in == "":
                return float(default)
            u_in = u_in.replace("%", "").strip()
            try:
                v = float(u_in)
                if min_val is not None and v < min_val:
                    print(f"   [!] ค่าต้องไม่ต่ำกว่า {min_val}")
                    continue
                if max_val is not None and v > max_val:
                    print(f"   [!] ค่าต้องไม่เกิน {max_val}")
                    continue
                return v
            except ValueError:
                print(f"   [!] ตัวเลขไม่ถูกต้อง กรุณากรอกใหม่")

    def prompt_choice(self, title, options, default_idx=1):
        print(f"\n[*] {title}:")
        for idx, opt in enumerate(options, 1):
            print(f"   [{idx}] {opt}")
        while True:
            u_in = input(f"   เลือกตัวเลือก [default: {default_idx} ({options[default_idx-1]})]: ").strip()
            if u_in == "":
                return options[default_idx-1]
            if u_in.isdigit():
                idx = int(u_in)
                if 1 <= idx <= len(options):
                    return options[idx-1]
            for opt in options:
                if u_in.lower() == opt.lower() or u_in.lower() == opt[0].lower():
                    return opt
            print(f"   [!] กรุณากรอกหมายเลข 1-{len(options)} หรือชื่อตัวเลือก")

    def classify_and_display(self, raw_inputs):
        vec = []
        for feat in self.feature_names:
            raw_v = raw_inputs.get(feat, 0)
            if feat in self.label_encoders:
                le = self.label_encoders[feat]
                str_v = str(raw_v)
                if str_v in le.classes_:
                    val = float(le.transform([str_v])[0])
                else:
                    val = 0.0
            else:
                val = self.scale_feature(feat, raw_v)
            vec.append(val)

        test_vec = np.array(vec)
        pred_cls, nn_idx, nn_dist, nn_classes, vote_counts = self.knn.predict_detailed(test_vec)

        print("\n" + "=" * 68)
        print(f" ผลการจำแนกข้อมูลด้วย K-NN (K = {self.knn.k})")
        print("=" * 68)
        print(f" >>> คลาสเป้าหมายที่จำแนกได้ (Predicted Class) : [  {pred_cls}  ] <<<")
        print("=" * 68)

        print(f"\n[ ตารางแสดง {self.knn.k} เพื่อนบ้านที่ใกล้ที่สุด (K-Nearest Neighbors) ]:")
        print(f"{'ลำดับ':<6} | {'Sample Index':<13} | {'Euclidean Distance':<18} | {'Class Label':<10}")
        print("-" * 55)
        for rank, (idx, dist, cls_name) in enumerate(zip(nn_idx, nn_dist, nn_classes), 1):
            print(f"{rank:<6} | #{idx:<12} | {dist:<18.4f} | Class '{cls_name}'")

        print("\n[ สรุปผลการโหวตจากเพื่อนบ้าน (Majority Voting Breakdown) ]:")
        total_votes = self.knn.k
        for cls_name, count in vote_counts.most_common():
            pct = (count / total_votes) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            star = " ★ (Winner)" if cls_name == pred_cls else ""
            print(f"   Class '{cls_name}' : [{bar}] {count}/{total_votes} votes ({pct:5.1f}%){star}")

        print("\n[ สรุปข้อมูลที่นำเข้า (Input Features) ]:")
        for k_in, v_in in raw_inputs.items():
            print(f"   - {k_in:24s}: {v_in}")

        print("\n[ เวกเตอร์ข้อมูลหลัง Normalize (0.0 - 1.0) ]:")
        for feat, val in zip(self.feature_names, vec):
            print(f"   - {feat:24s}: {val:.4f}")
        print("=" * 68)

    def interactive_input(self):
        print("\n" + "-" * 68)
        print(" กรุณากรอกข้อมูลทดสอบเพื่อจำแนกกลุ่ม (INPUT TEST DATA)")
        print("-" * 68)
        inputs = {}

        if 'study_hours' in self.feature_names:
            inputs['study_hours'] = self.prompt_number(
                "ชั่วโมงการอ่านหนังสือ/เรียนต่อวัน (Daily Study Hours)",
                "จำนวนชั่วโมงต่อวัน (ช่วง 1 - 12 ชม.)", default=6.0, min_val=0.0, max_val=24.0
            )
        if 'attendance' in self.feature_names:
            inputs['attendance'] = self.prompt_number(
                "อัตราการเข้าชั้นเรียน (Attendance Rate %)",
                "เปอร์เซ็นต์การเข้าเรียน (ช่วง 40% - 100%)", default=85.0, min_val=0.0, max_val=100.0
            )
        if 'sleep_hours' in self.feature_names:
            inputs['sleep_hours'] = self.prompt_number(
                "ชั่วโมงการนอนหลับเฉลี่ย (Daily Sleep Hours)",
                "จำนวนชั่วโมงนอนต่อคืน (ช่วง 4 - 10 ชม.)", default=7.0, min_val=0.0, max_val=24.0
            )
        if 'internet_usage' in self.feature_names:
            inputs['internet_usage'] = self.prompt_number(
                "ชั่วโมงการใช้งานอินเทอร์เน็ตทั่วไป (Internet Usage)",
                "จำนวนชั่วโมงต่อวัน (ช่วง 1 - 8 ชม.)", default=3.0, min_val=0.0, max_val=24.0
            )
        if 'assignments_completed' in self.feature_names:
            inputs['assignments_completed'] = self.prompt_number(
                "จำนวนการบ้าน/งานที่ส่งครบ (Assignments Completed)",
                "จำนวนชิ้นงานที่ส่ง (ช่วง 0 - 10 ชิ้น)", default=8.0, min_val=0.0, max_val=20.0
            )
        if 'previous_score' in self.feature_names:
            inputs['previous_score'] = self.prompt_number(
                "คะแนนสอบเดิม (Previous Exam Score)",
                "คะแนนสอบครั้งก่อน (ช่วง 35 - 95 คะแนน)", default=78.0, min_val=0.0, max_val=100.0
            )
            score = inputs['previous_score']
            if score >= 80: auto_g = 'A'
            elif score >= 70: auto_g = 'B'
            elif score >= 60: auto_g = 'C'
            elif score >= 50: auto_g = 'D'
            else: auto_g = 'F'
        else:
            auto_g = 'B'

        if 'PreviousGrade' in self.feature_names:
            inputs['PreviousGrade'] = self.prompt_choice(
                "เกรดเฉลี่ยเดิม (Previous Academic Grade)",
                ["A", "B", "C", "D", "F"],
                default_idx=['A', 'B', 'C', 'D', 'F'].index(auto_g) + 1
            )

        if 'Gender' in self.feature_names:
            inputs['Gender'] = self.prompt_choice("เพศ (Gender)", ["Male", "Female"], default_idx=1)
        if 'ParentalSupport' in self.feature_names:
            inputs['ParentalSupport'] = self.prompt_choice("ระดับการสนับสนุนของผู้ปกครอง (Parental Support)", ["High", "Medium", "Low"], default_idx=1)

        self.classify_and_display(inputs)

    def run_presets(self):
        presets = [
            ("นักเรียนกลุ่มผลการเรียนดีเด่น (Top-Performing Student)", {
                'study_hours': 10.0, 'attendance': 95.0, 'sleep_hours': 8.0,
                'internet_usage': 2.0, 'assignments_completed': 10.0,
                'previous_score': 92.0, 'PreviousGrade': 'A',
                'Gender': 'Female', 'ParentalSupport': 'High'
            }),
            ("นักเรียนกลุ่มผลการเรียนปานกลาง (Average Student)", {
                'study_hours': 5.0, 'attendance': 75.0, 'sleep_hours': 7.0,
                'internet_usage': 4.0, 'assignments_completed': 6.0,
                'previous_score': 68.0, 'PreviousGrade': 'C',
                'Gender': 'Male', 'ParentalSupport': 'Medium'
            }),
            ("นักเรียนกลุ่มเสี่ยงตก/ต้องปรับปรุง (At-Risk Student)", {
                'study_hours': 1.5, 'attendance': 45.0, 'sleep_hours': 4.0,
                'internet_usage': 7.0, 'assignments_completed': 2.0,
                'previous_score': 40.0, 'PreviousGrade': 'F',
                'Gender': 'Male', 'ParentalSupport': 'Low'
            })
        ]

        print("\n" + "=" * 68)
        print(f" ทดสอบจำแนกข้อมูลตัวอย่างกรณีศึกษา (K = {self.knn.k})")
        print("=" * 68)
        for title, profile in presets:
            print(f"\n>>> กำลังทดสอบ: {title}")
            self.classify_and_display(profile)

    def change_k(self):
        new_k = self.prompt_number(
            "กำหนดค่า K ใหม่ (Number of Nearest Neighbors)",
            f"ระบุจำนวนเพื่อนบ้าน K (ปัจจุบัน K = {self.knn.k})",
            default=self.knn.k, min_val=1, max_val=100
        )
        self.k = int(new_k)
        self.knn.k = self.k
        print(f"\n[✓] เปลี่ยนค่า K สำเร็จ: ปัจจุบันใช้ K = {self.knn.k}")

    def run(self):
        self.load_and_train()
        while True:
            print("\n" + "=" * 68)
            print(f" เมนูหลัก - K-NN CLASSIFIER (ปัจจุบัน K = {self.knn.k})")
            print("=" * 68)
            print("  [1] กรอกข้อมูลทดสอบใหม่ (Input Test Data for Classification)")
            print("  [2] ทดสอบข้อมูลกรณีศึกษาตัวอย่าง (Run Preset Test Profiles)")
            print("  [3] ประเมินความแม่นยำของโมเดลบน Test Set (Evaluate Accuracy & Confusion Matrix)")
            print("  [4] เปลี่ยนค่า K (Change K Value)")
            print("  [5] ออกจากโปรแกรม (Exit)")
            print("=" * 68)

            choice = input("เลือกเมนู [1-5] (default: 1): ").strip()
            if choice == "" or choice == "1":
                self.interactive_input()
            elif choice == "2":
                self.run_presets()
            elif choice == "3":
                self.evaluate_model()
            elif choice == "4":
                self.change_k()
            elif choice == "5" or choice.lower() in ['exit', 'quit', 'q']:
                print("\n[✓] ขอบคุณที่ใช้งานโปรแกรม K-NN Classification. สวัสดีครับ!\n")
                break
            else:
                print("[!] เมนูไม่ถูกต้อง กรุณากรอก 1 - 5")

def main():
    app = Lab4KNNApp(default_k=5)
    if '--preset' in sys.argv:
        app.load_and_train()
        app.run_presets()
    elif '--eval' in sys.argv:
        app.load_and_train()
        app.evaluate_model()
    else:
        app.run()

if __name__ == '__main__':
    main()
