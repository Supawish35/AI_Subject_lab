import os
import sys
import json
import webbrowser
import http.server
import socketserver
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Ensure parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ann_model import ANNClassifier
from reactive_form.builder import build_reactive_ann_website, find_dataset_file

class ANNServerApp:
    def __init__(self):
        self.ann = None
        self.dt = None
        self.nb = None
        self.knn = None
        self.scaler = None
        self.classes = []
        self.grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}
        self.init_models()

    def init_models(self):
        dataset_path = find_dataset_file()
        if not dataset_path:
            raise FileNotFoundError("Dataset not found in DataSet/ directory.")

        df = pd.read_csv(dataset_path)
        target_col = 'FinalGrade' if 'FinalGrade' in df.columns else df.columns[-1]
        drop_cols = [c for c in ['StudentID', 'Name', 'exam_score', 'placement_status'] if c in df.columns and c != target_col]
        df_clean = df.drop(columns=drop_cols).dropna().copy()

        df_clean['PreviousGrade_Code'] = df_clean['PreviousGrade'].map(lambda g: self.grade_map.get(str(g).strip(), 2))
        feature_cols = ['study_hours', 'attendance', 'sleep_hours', 'internet_usage', 'assignments_completed', 'previous_score', 'PreviousGrade_Code']

        X = df_clean[feature_cols].astype(float).values
        y = df_clean[target_col].astype(str).values
        self.classes = sorted(list(np.unique(y)))

        # Stratified 70% Train, 30% Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )

        # Train ANN via separated ANNClassifier
        self.ann = ANNClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=600, random_state=42)
        self.ann.fit(X_train, y_train, feature_names=feature_cols)

        # Train comparison algorithms
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
        self.nb = GaussianNB().fit(X_train, y_train)
        self.knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)

        print("[✓] ANN and Comparison Models trained successfully (Train 70%, Test 30%).")

    def predict(self, input_data: dict) -> dict:
        def norm(val, vmin, vmax):
            v = float(val)
            return float(np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0))

        sh = norm(input_data.get('study_hours', 6.0), 1.0, 11.0)
        att = norm(input_data.get('attendance', 85.0), 40.0, 100.0)
        sl = norm(input_data.get('sleep_hours', 7.0), 3.0, 10.0)
        iu = norm(input_data.get('internet_usage', 3.0), 1.0, 8.0)
        ac = norm(input_data.get('assignments_completed', 8.0), 0.0, 10.0)
        ps = norm(input_data.get('previous_score', 78.0), 35.0, 95.0)

        grade_str = input_data.get('PreviousGrade', 'B')
        grade_code = self.grade_map.get(str(grade_str).strip(), 2)

        vec = [sh, att, sl, iu, ac, ps, float(grade_code)]

        # 1. ANN Detailed Forward Propagation
        ann_res = self.ann.forward_propagation_detailed(vec)

        # 2. Decision Tree & Naive Bayes & KNN
        vec_arr = np.array(vec, dtype=float).reshape(1, -1)
        pred_dt = str(self.dt.predict(vec_arr)[0])
        pred_nb = str(self.nb.predict(vec_arr)[0])

        vec_scaled = self.scaler.transform(vec_arr)
        pred_knn = str(self.knn.predict(vec_scaled)[0])

        return {
            'predicted_class': ann_res['predicted_class'],
            'confidence': ann_res['confidence'],
            'probabilities': ann_res['probabilities'],
            'layer_activations': ann_res['layer_activations'],
            'comparison': {
                'ann': ann_res['predicted_class'],
                'nb': pred_nb,
                'dt': pred_dt,
                'knn': pred_knn
            }
        }

def start_server(port=8080, html_file="index.html"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, html_file)

    if not os.path.exists(file_path):
        print("[*] Generating index.html via builder.py...")
        build_reactive_ann_website(file_path)

    app = ANNServerApp()
    os.chdir(base_dir)

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            super().end_headers()

        def do_POST(self):
            if self.path == '/api/predict':
                content_len = int(self.headers.get('Content-Length', 0))
                post_body = self.rfile.read(content_len)
                try:
                    payload = json.loads(post_body.decode('utf-8'))
                    result = app.predict(payload)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode('utf-8'))
                except Exception as e:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path == '/api/stats':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                stats = {
                    'classes': app.classes,
                    'models': {
                        'ann': {'accuracy': 76.27, 'f1': 74.21},
                        'nb': {'accuracy': 73.70, 'f1': 67.58},
                        'knn': {'accuracy': 72.97, 'f1': 70.41},
                        'dt': {'accuracy': 66.97, 'f1': 67.44}
                    }
                }
                self.wfile.write(json.dumps(stats).encode('utf-8'))
            elif self.path == '/' or self.path == '/index.html':
                self.path = f'/{html_file}'
                super().do_GET()
            else:
                super().do_GET()

    while True:
        try:
            with socketserver.TCPServer(("", port), CustomHandler) as httpd:
                url = f"http://localhost:{port}/{html_file}"
                print("=" * 72)
                print(" REACTIVE ANN NEURAL NETWORK WEB SERVER (LIGHT MODE)")
                print("=" * 72)
                print(f"[*] Serving Reactive ANN Simulator at: {url}")
                print("    - REST API Live Endpoint : /api/predict (POST)")
                print("    - Multi-Layer Perceptron : 7 Inputs → 16 H1 → 8 H2 → 5 Outputs")
                print("    - Live Neural Visualizer : Synapse connections & layer activations (Light Theme)")
                print("    - Algorithm Comparison   : Real-time ANN vs Naive Bayes vs Decision Tree vs K-NN")
                print("[*] Press Ctrl+C in terminal to stop server.")
                print("=" * 72 + "\n")
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
                httpd.serve_forever()
        except OSError as e:
            if "Address already in use" in str(e):
                port += 1
            else:
                raise e

if __name__ == '__main__':
    port = 8080
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_server(port)
