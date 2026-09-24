import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def find_file(relative_paths):
    """Locate existing file from a list of candidate relative paths."""
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

class NaiveBayesPredictorApp:
    def __init__(self):
        self.model = GaussianNB()
        self.feature_names = []
        self.label_encoders = {}
        self.raw_scalers = {}
        self.train_acc = 0.0
        self.test_acc = 0.0
        self.classes = []

    def train(self):
        dataset_path = find_file([
            '../DataSet/student_dataset_full_normalized.csv',
            'DataSet/student_dataset_full_normalized.csv',
            '../DataSet/student_performance_normalized.csv',
            'DataSet/student_performance_normalized.csv'
        ])

        if not dataset_path:
            # Auto generate full dataset if missing
            try:
                from normalize import normalize_10k_dataset
                normalize_10k_dataset()
                dataset_path = find_file(['../DataSet/student_dataset_full_normalized.csv', 'DataSet/student_dataset_full_normalized.csv'])
            except Exception:
                pass

        if not dataset_path or not os.path.exists(dataset_path):
            print("[!] Error: Could not locate normalized dataset. Please run normalize.py first.")
            sys.exit(1)

        print("=" * 68)
        print(" STUDENT GRADE PREDICTOR CLI (NAIVE BAYES CLASSIFICATION)")
        print("=" * 68)
        print(f"[*] Loading training dataset: {os.path.basename(dataset_path)}...")

        df = pd.read_csv(dataset_path)
        target_col = 'FinalGrade' if 'FinalGrade' in df.columns else df.columns[-1]

        drop_cols = [c for c in ['StudentID', 'Name', 'exam_score', 'placement_status'] if c in df.columns and c != target_col]
        df_clean = df.drop(columns=drop_cols).dropna()

        X = df_clean.drop(columns=[target_col]).copy()
        y = df_clean[target_col].astype(str)

        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object' or str(X[col].dtype).startswith('str'):
                le = LabelEncoder()
                if col == 'PreviousGrade':
                    all_grades = ['A', 'B', 'C', 'D', 'F']
                    le.fit(all_grades)
                    X[col] = le.transform(X[col].astype(str))
                else:
                    X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le

        X = X.astype(float)
        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        self.classes = sorted(list(self.model.classes_))
        self.train_acc = accuracy_score(y_train, self.model.predict(X_train))
        self.test_acc = accuracy_score(y_test, self.model.predict(X_test))

        # Learn raw feature scaling bounds
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
        
        # Fallback default scalers if raw file not directly available
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
        for k, v in default_bounds.items():
            if k not in self.raw_scalers:
                self.raw_scalers[k] = v

        print(f"[*] Model trained successfully on {len(X_train):,} samples.")
        print(f"    - Training Accuracy: {self.train_acc * 100:.2f}%")
        print(f"    - Testing Accuracy : {self.test_acc * 100:.2f}%")
        print(f"    - Target Classes   : {', '.join(self.classes)}")
        print("=" * 68)

    def scale_value(self, feature_name, val):
        if feature_name in self.raw_scalers:
            min_v, max_v = self.raw_scalers[feature_name]
            if max_v == min_v:
                return 0.5
            clamped = max(min_v, min(max_v, float(val)))
            return (clamped - min_v) / (max_v - min_v)
        return float(val)

    def prompt_choice(self, title, options, default_idx=1):
        print(f"\n[*] {title}:")
        for idx, opt in enumerate(options, 1):
            print(f"   [{idx}] {opt}")
        while True:
            user_in = input(f"   Select option [default: {default_idx} ({options[default_idx-1]})]: ").strip()
            if user_in == "":
                return options[default_idx-1]
            if user_in.isdigit():
                idx = int(user_in)
                if 1 <= idx <= len(options):
                    return options[idx-1]
            for opt in options:
                if user_in.lower() == opt.lower() or user_in.lower() == opt[0].lower():
                    return opt
            print(f"   [!] Invalid input '{user_in}'. Please choose a valid number or option.")

    def prompt_number(self, title, hint, default, min_val=None, max_val=None):
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
                print(f"   [!] Invalid number '{user_in}'. Please try again.")

    def predict_instance(self, raw_inputs):
        model_input = {}
        for feat in self.feature_names:
            raw_v = raw_inputs.get(feat, 0)
            if feat in self.label_encoders:
                le = self.label_encoders[feat]
                str_v = str(raw_v)
                if str_v in le.classes_:
                    model_input[feat] = float(le.transform([str_v])[0])
                else:
                    model_input[feat] = 0.0
            else:
                model_input[feat] = self.scale_value(feat, raw_v)

        sample_df = pd.DataFrame([model_input])[self.feature_names]
        predicted_grade = self.model.predict(sample_df)[0]
        probabilities = self.model.predict_proba(sample_df)[0]
        
        prob_dict = dict(zip(self.model.classes_, probabilities))
        return predicted_grade, prob_dict, model_input

    def display_prediction(self, raw_inputs, predicted_grade, prob_dict, model_input):
        print("\n" + "=" * 68)
        print(" NAIVE BAYES PREDICTION RESULT")
        print("=" * 68)
        print(f" >>> PREDICTED FINAL GRADE :  [  {predicted_grade}  ] <<<")
        print("=" * 68)

        print("\n[ Posterior Probability Distribution P(Grade | Input Features) ]:")
        for cls_name in sorted(prob_dict.keys()):
            prob = prob_dict[cls_name]
            percentage = prob * 100
            bar_len = int(percentage / 2.5)  # 40 chars = 100%
            bar = "█" * bar_len + "░" * (40 - bar_len)
            star = " ★ (Predicted)" if cls_name == predicted_grade else ""
            print(f"   Grade {cls_name} : [{bar}] {percentage:6.2f}%{star}")

        print("\n[ Summary of Student Input Features ]:")
        for k, v in raw_inputs.items():
            print(f"   - {k:24s}: {v}")

        print("\n[ Normalized Model Vectors (0.0 - 1.0) ]:")
        for k, v in model_input.items():
            print(f"   - {k:24s}: {v:.4f}")
        print("=" * 68)

    def interactive_input(self):
        print("\n" + "-" * 68)
        print(" ENTER STUDENT DETAILS FOR PREDICTION")
        print("-" * 68)
        inputs = {}

        if 'study_hours' in self.feature_names:
            inputs['study_hours'] = self.prompt_number(
                "Daily Study Hours", "Hours spent studying per day [Range: 1 - 12 hours]", default=6.0, min_val=0.0, max_val=24.0
            )
        if 'attendance' in self.feature_names:
            inputs['attendance'] = self.prompt_number(
                "Class Attendance Rate", "Attendance percentage [Range: 40% - 100%]", default=85.0, min_val=0.0, max_val=100.0
            )
        if 'sleep_hours' in self.feature_names:
            inputs['sleep_hours'] = self.prompt_number(
                "Daily Sleep Hours", "Average hours of sleep per night [Range: 4 - 10 hours]", default=7.0, min_val=0.0, max_val=24.0
            )
        if 'internet_usage' in self.feature_names:
            inputs['internet_usage'] = self.prompt_number(
                "Internet Usage Hours", "Daily non-study internet usage [Range: 1 - 8 hours]", default=3.0, min_val=0.0, max_val=24.0
            )
        if 'assignments_completed' in self.feature_names:
            inputs['assignments_completed'] = self.prompt_number(
                "Assignments Completed", "Number of assignments turned in [Range: 0 - 10]", default=8.0, min_val=0.0, max_val=20.0
            )
        if 'previous_score' in self.feature_names:
            inputs['previous_score'] = self.prompt_number(
                "Previous Exam Score", "Previous exam score (0 - 100) [Range: 35 - 95]", default=78.0, min_val=0.0, max_val=100.0
            )
            # Automatic grade inference from score
            score = inputs['previous_score']
            if score >= 80: prev_g = 'A'
            elif score >= 70: prev_g = 'B'
            elif score >= 60: prev_g = 'C'
            elif score >= 50: prev_g = 'D'
            else: prev_g = 'F'
        else:
            prev_g = 'B'

        if 'PreviousGrade' in self.feature_names:
            inputs['PreviousGrade'] = self.prompt_choice(
                "Previous Academic Grade", ["A", "B", "C", "D", "F"],
                default_idx=['A', 'B', 'C', 'D', 'F'].index(prev_g) + 1
            )

        # Support 1k-dataset features if loaded
        if 'Gender' in self.feature_names:
            inputs['Gender'] = self.prompt_choice("Gender", ["Male", "Female"], default_idx=1)
        if 'ParentalSupport' in self.feature_names:
            inputs['ParentalSupport'] = self.prompt_choice("Parental Support", ["High", "Medium", "Low"], default_idx=1)

        pred_grade, probs, model_vec = self.predict_instance(inputs)
        self.display_prediction(inputs, pred_grade, probs, model_vec)

    def run_presets(self):
        presets = [
            ("Top-Performing Student (นักเรียนระดับดีเลิศ)", {
                'study_hours': 10.0, 'attendance': 96.0, 'sleep_hours': 8.0,
                'internet_usage': 2.0, 'assignments_completed': 10.0,
                'previous_score': 92.0, 'PreviousGrade': 'A',
                'Gender': 'Female', 'ParentalSupport': 'High'
            }),
            ("Average Student (นักเรียนระดับปานกลาง)", {
                'study_hours': 5.0, 'attendance': 75.0, 'sleep_hours': 7.0,
                'internet_usage': 4.0, 'assignments_completed': 6.0,
                'previous_score': 68.0, 'PreviousGrade': 'C',
                'Gender': 'Male', 'ParentalSupport': 'Medium'
            }),
            ("At-Risk Student (นักเรียนกลุ่มเสี่ยงติด F/D)", {
                'study_hours': 1.5, 'attendance': 45.0, 'sleep_hours': 4.0,
                'internet_usage': 7.0, 'assignments_completed': 2.0,
                'previous_score': 42.0, 'PreviousGrade': 'F',
                'Gender': 'Male', 'ParentalSupport': 'Low'
            })
        ]

        print("\n" + "=" * 68)
        print(" RUNNING PRESET STUDENT PROFILES")
        print("=" * 68)
        for label, profile in presets:
            print(f"\n>>> Profile: {label}")
            pred_grade, probs, model_vec = self.predict_instance(profile)
            self.display_prediction(profile, pred_grade, probs, model_vec)

    def run(self):
        self.train()
        while True:
            print("\n" + "=" * 68)
            print(" MAIN MENU - SELECT ACTION")
            print("=" * 68)
            print("  [1] Enter Custom Student Info (กรอกข้อมูลนักเรียนใหม่)")
            print("  [2] Run Preset Student Profiles (ทดสอบกรณีศึกษาตัวอย่าง)")
            print("  [3] Exit Program (ออกจากโปรแกรม)")
            print("=" * 68)

            choice = input("Enter option [1-3] (default: 1): ").strip()
            if choice == "" or choice == "1":
                self.interactive_input()
            elif choice == "2":
                self.run_presets()
            elif choice == "3" or choice.lower() in ['exit', 'quit', 'q']:
                print("\n[✓] Thank you for using Student Grade Predictor. Goodbye!\n")
                break
            else:
                print("[!] Invalid option. Please enter 1, 2, or 3.")

def main():
    if '--preset' in sys.argv:
        app = NaiveBayesPredictorApp()
        app.train()
        app.run_presets()
    else:
        app = NaiveBayesPredictorApp()
        app.run()

if __name__ == '__main__':
    main()
