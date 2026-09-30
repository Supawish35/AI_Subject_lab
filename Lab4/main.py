import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from knn_model import KNNClassifier
from report_generator import generate_html_report_file
from web_builder import build_reactive_website

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
        self.dataset_name = ""
        self.dataset_path = ""
        self.eval_results = None
        self.preset_results = []
        self.last_custom_result = None

    def load_and_train(self):
        # Prioritize 10,000-row full normalized dataset
        dataset_path = find_file([
            '../DataSet/student_dataset_full_normalized.csv',
            'DataSet/student_dataset_full_normalized.csv',
            '../DataSet/student_performance_normalized.csv',
            'DataSet/student_performance_normalized.csv'
        ])

        if not dataset_path:
            print("[*] Normalized dataset not found, generating via Lab3 normalize pipeline...")
            try:
                lab3_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Lab3'))
                if lab3_dir not in sys.path:
                    sys.path.append(lab3_dir)
                from normalize import normalize_10k_dataset
                normalize_10k_dataset()
                dataset_path = find_file(['../DataSet/student_dataset_full_normalized.csv', 'DataSet/student_dataset_full_normalized.csv'])
            except Exception as err:
                print(f"[!] Warning: Auto-normalization attempt failed: {err}")

        if not dataset_path or not os.path.exists(dataset_path):
            print("[!] Error: Could not locate normalized dataset file in DataSet directory.")
            sys.exit(1)

        self.dataset_path = dataset_path
        self.dataset_name = os.path.basename(dataset_path)

        print("=" * 68)
        print(" LAB 4: K-NEAREST NEIGHBORS (K-NN) CLASSIFICATION")
        print("=" * 68)
        print(f"[*] Loading dataset: {self.dataset_name}")

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

        # Train / Test split (80:20 stratified split)
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

        print(f"[*] K-NN model initialized (Default K = {self.k})")
        print(f"    - Training Samples : {len(self.X_train):,} rows")
        print(f"    - Testing Samples  : {len(self.X_test):,} rows")
        print(f"    - Total Rows       : {len(self.X_train) + len(self.X_test):,} rows")
        print(f"    - Features ({len(self.feature_names)}): {', '.join(self.feature_names)}")
        print(f"    - Target Classes   : {', '.join(self.classes)}")
        print("=" * 68)

    def evaluate_model(self, auto_export_html=True):
        print("\n" + "=" * 68)
        print(f" MODEL PERFORMANCE EVALUATION (K = {self.k})")
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

        rep_dict = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
        self.eval_results = {
            'accuracy': float(acc),
            'confusion_matrix': cm.tolist(),
            'classes': list(self.classes),
            'report_dict': rep_dict,
            'test_samples': len(self.X_test),
            'train_samples': len(self.X_train),
            'k': self.knn.k
        }

        if auto_export_html:
            self.generate_html_report()

        return self.eval_results

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
            u_in = input(f"   Enter value [default: {default}]: ").strip()
            if u_in == "":
                return float(default)
            u_in = u_in.replace("%", "").strip()
            try:
                v = float(u_in)
                if min_val is not None and v < min_val:
                    print(f"   [!] Value cannot be less than {min_val}")
                    continue
                if max_val is not None and v > max_val:
                    print(f"   [!] Value cannot exceed {max_val}")
                    continue
                return v
            except ValueError:
                print(f"   [!] Invalid number. Please enter a numeric value.")

    def prompt_choice(self, title, options, default_idx=1):
        print(f"\n[*] {title}:")
        for idx, opt in enumerate(options, 1):
            print(f"   [{idx}] {opt}")
        while True:
            u_in = input(f"   Select option [default: {default_idx} ({options[default_idx-1]})]: ").strip()
            if u_in == "":
                return options[default_idx-1]
            if u_in.isdigit():
                idx = int(u_in)
                if 1 <= idx <= len(options):
                    return options[idx-1]
            for opt in options:
                if u_in.lower() == opt.lower() or u_in.lower() == opt[0].lower():
                    return opt
            print(f"   [!] Invalid selection. Please enter 1-{len(options)} or option name.")

    def classify_and_display(self, raw_inputs, profile_title="Student Test Profile"):
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
        print(f" K-NN CLASSIFICATION RESULT (K = {self.knn.k})")
        print("=" * 68)
        print(f" >>> PREDICTED TARGET CLASS : [  {pred_cls}  ] <<<")
        print("=" * 68)

        print(f"\n[ Nearest Neighbors (K = {self.knn.k}) ]:")
        print(f"{'Rank':<6} | {'Sample Index':<13} | {'Euclidean Distance':<18} | {'Class Label':<10}")
        print("-" * 55)
        for rank, (idx, dist, cls_name) in enumerate(zip(nn_idx, nn_dist, nn_classes), 1):
            print(f"{rank:<6} | #{idx:<12} | {dist:<18.4f} | Class '{cls_name}'")

        print("\n[ Majority Voting Breakdown ]:")
        total_votes = self.knn.k
        for cls_name, count in vote_counts.most_common():
            pct = (count / total_votes) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            star = " ★ (Winner)" if cls_name == pred_cls else ""
            print(f"   Class '{cls_name}' : [{bar}] {count}/{total_votes} votes ({pct:5.1f}%){star}")

        print("\n[ Summary of Input Features ]:")
        for k_in, v_in in raw_inputs.items():
            print(f"   - {k_in:24s}: {v_in}")

        print("\n[ Normalized Feature Vector (0.0 - 1.0) ]:")
        for feat, val in zip(self.feature_names, vec):
            print(f"   - {feat:24s}: {val:.4f}")
        print("=" * 68)

        result_payload = {
            'title': profile_title,
            'raw_inputs': raw_inputs,
            'normalized_vector': dict(zip(self.feature_names, [round(float(v), 4) for v in vec])),
            'predicted_class': str(pred_cls),
            'nearest_neighbors': [
                {'rank': r, 'index': int(idx), 'distance': round(float(dist), 4), 'class': str(c)}
                for r, (idx, dist, c) in enumerate(zip(nn_idx, nn_dist, nn_classes), 1)
            ],
            'vote_counts': {str(c): int(count) for c, count in vote_counts.most_common()},
            'total_votes': int(self.knn.k),
            'k': self.knn.k
        }
        return result_payload

    def interactive_input(self):
        print("\n" + "-" * 68)
        print(" INPUT TEST DATA FOR CLASSIFICATION")
        print("-" * 68)
        inputs = {}

        if 'study_hours' in self.feature_names:
            inputs['study_hours'] = self.prompt_number(
                "Daily Study Hours",
                "Hours spent studying per day (Range: 1 - 12 hrs)", default=6.0, min_val=0.0, max_val=24.0
            )
        if 'attendance' in self.feature_names:
            inputs['attendance'] = self.prompt_number(
                "Class Attendance Rate (%)",
                "Attendance percentage (Range: 40% - 100%)", default=85.0, min_val=0.0, max_val=100.0
            )
        if 'sleep_hours' in self.feature_names:
            inputs['sleep_hours'] = self.prompt_number(
                "Daily Sleep Hours",
                "Average hours of sleep per night (Range: 4 - 10 hrs)", default=7.0, min_val=0.0, max_val=24.0
            )
        if 'internet_usage' in self.feature_names:
            inputs['internet_usage'] = self.prompt_number(
                "Internet Usage Hours",
                "Non-academic internet usage hours (Range: 1 - 8 hrs)", default=3.0, min_val=0.0, max_val=24.0
            )
        if 'assignments_completed' in self.feature_names:
            inputs['assignments_completed'] = self.prompt_number(
                "Assignments Completed",
                "Total assignments completed (Range: 0 - 10)", default=8.0, min_val=0.0, max_val=20.0
            )
        if 'previous_score' in self.feature_names:
            inputs['previous_score'] = self.prompt_number(
                "Previous Exam Score",
                "Score in previous test (Range: 35 - 95)", default=78.0, min_val=0.0, max_val=100.0
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
                "Previous Academic Grade",
                ["A", "B", "C", "D", "F"],
                default_idx=['A', 'B', 'C', 'D', 'F'].index(auto_g) + 1
            )

        if 'Gender' in self.feature_names:
            inputs['Gender'] = self.prompt_choice("Gender", ["Male", "Female"], default_idx=1)
        if 'ParentalSupport' in self.feature_names:
            inputs['ParentalSupport'] = self.prompt_choice("Parental Support", ["High", "Medium", "Low"], default_idx=1)

        result_payload = self.classify_and_display(inputs, profile_title="Interactive User Query")
        self.last_custom_result = result_payload

    def run_presets(self, auto_export_html=True):
        presets = [
            ("Top-Performing Student", {
                'study_hours': 10.0, 'attendance': 95.0, 'sleep_hours': 8.0,
                'internet_usage': 2.0, 'assignments_completed': 10.0,
                'previous_score': 92.0, 'PreviousGrade': 'A',
                'Gender': 'Female', 'ParentalSupport': 'High'
            }),
            ("Average Student", {
                'study_hours': 5.0, 'attendance': 75.0, 'sleep_hours': 7.0,
                'internet_usage': 4.0, 'assignments_completed': 6.0,
                'previous_score': 68.0, 'PreviousGrade': 'C',
                'Gender': 'Male', 'ParentalSupport': 'Medium'
            }),
            ("At-Risk Student", {
                'study_hours': 1.5, 'attendance': 45.0, 'sleep_hours': 4.0,
                'internet_usage': 7.0, 'assignments_completed': 2.0,
                'previous_score': 40.0, 'PreviousGrade': 'F',
                'Gender': 'Male', 'ParentalSupport': 'Low'
            })
        ]

        print("\n" + "=" * 68)
        print(f" RUNNING BENCHMARK PRESET CASE STUDIES (K = {self.knn.k})")
        print("=" * 68)
        self.preset_results = []
        for title, profile in presets:
            print(f"\n>>> Profile: {title}")
            res = self.classify_and_display(profile, profile_title=title)
            self.preset_results.append(res)

        if auto_export_html:
            self.generate_html_report()

    def change_k(self):
        new_k = self.prompt_number(
            "Change K Value (Number of Nearest Neighbors)",
            f"Enter new K value (Current K = {self.knn.k})",
            default=self.knn.k, min_val=1, max_val=100
        )
        self.k = int(new_k)
        self.knn.k = self.k
        print(f"\n[✓] K value updated: Now using K = {self.knn.k}")

    def generate_html_report(self, output_path=None):
        if self.eval_results is None:
            y_pred = self.knn.predict(self.X_test)
            acc = float(accuracy_score(self.y_test, y_pred))
            cm = confusion_matrix(self.y_test, y_pred, labels=self.classes)
            rep_dict = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
            self.eval_results = {
                'accuracy': acc,
                'confusion_matrix': cm.tolist(),
                'classes': list(self.classes),
                'report_dict': rep_dict,
                'test_samples': len(self.X_test),
                'train_samples': len(self.X_train),
                'k': self.knn.k
            }

        if not self.preset_results:
            self.run_presets(auto_export_html=False)

        return generate_html_report_file(
            eval_results=self.eval_results,
            preset_results=self.preset_results,
            last_custom_result=self.last_custom_result,
            feature_names=self.feature_names,
            dataset_name=self.dataset_name,
            current_k=self.knn.k,
            output_path=output_path
        )

    def launch_reactive_website(self, port=8080):
        """Compile and serve the reactive website."""
        out_html = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')
        build_reactive_website(self.dataset_path, out_html)
        print(f"\n[✓] Reactive website created at: {out_html}")
        print(f"[*] Starting local server on port {port}...")
        from server import start_server
        start_server(port, 'index.html')

    def run(self):
        self.load_and_train()
        while True:
            print("\n" + "=" * 68)
            print(f" MAIN MENU - K-NN CLASSIFIER (Current K = {self.knn.k})")
            print("=" * 68)
            print("  [1] Enter Custom Student Info (Interactive Classification)")
            print("  [2] Run Preset Student Profiles (Case Studies)")
            print("  [3] Evaluate Accuracy & Confusion Matrix on Test Set")
            print("  [4] Generate Static HTML Report")
            print("  [5] Launch Reactive K-NN Model Website (Live Sliders & 2D Projection)")
            print("  [6] Change K Value")
            print("  [7] Exit Program")
            print("=" * 68)

            choice = input("Enter option [1-7] (default: 1): ").strip()
            if choice == "" or choice == "1":
                self.interactive_input()
            elif choice == "2":
                self.run_presets()
            elif choice == "3":
                self.evaluate_model()
            elif choice == "4":
                self.generate_html_report()
            elif choice == "5":
                self.launch_reactive_website()
            elif choice == "6":
                self.change_k()
            elif choice == "7" or choice.lower() in ['exit', 'quit', 'q']:
                print("\n[✓] Thank you for using K-NN Classifier. Goodbye!\n")
                break
            else:
                print("[!] Invalid option. Please enter 1 - 7.")

def main():
    app = Lab4KNNApp(default_k=5)
    if '--preset' in sys.argv:
        app.load_and_train()
        app.run_presets()
    elif '--eval' in sys.argv:
        app.load_and_train()
        app.evaluate_model()
    elif '--html' in sys.argv:
        app.load_and_train()
        app.evaluate_model(auto_export_html=False)
        app.run_presets(auto_export_html=False)
        app.generate_html_report()
    elif '--web' in sys.argv:
        app.load_and_train()
        app.launch_reactive_website()
    else:
        app.run()

if __name__ == '__main__':
    main()
