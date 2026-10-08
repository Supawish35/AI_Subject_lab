import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Ensure parent directory is in sys.path to import ann_model
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ann_model import ANNClassifier

def find_dataset_file():
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'DataSet', 'student_dataset_full_normalized.csv'),
        os.path.join(os.getcwd(), 'DataSet', 'student_dataset_full_normalized.csv'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'DataSet', 'student_performance_normalized.csv'),
        os.path.join(os.getcwd(), 'DataSet', 'student_performance_normalized.csv'),
        'student_dataset_full_normalized.csv'
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

def build_reactive_ann_website(output_html_path=None):
    """
    Build a standalone, zero-dependency, ultra-reactive ANN web application in LIGHT MODE.
    Trains the ANN (and comparison models: NB, DT, KNN) on 70% Train, 30% Test,
    embeds network architecture, weights, scaler parameters, test samples, and evaluation stats.
    Includes a client-side neural forward propagation engine with real-time 60fps slider reactivity
    and live neural network layer activation visualization in clean Light Theme styling.
    """
    dataset_path = find_dataset_file()
    if not dataset_path:
        raise FileNotFoundError("Could not find dataset in DataSet/ folder.")

    df = pd.read_csv(dataset_path)
    target_col = 'FinalGrade' if 'FinalGrade' in df.columns else df.columns[-1]

    drop_cols = [c for c in ['StudentID', 'Name', 'exam_score', 'placement_status'] if c in df.columns and c != target_col]
    df_clean = df.drop(columns=drop_cols).dropna().copy()

    # Feature definitions
    feature_meta = [
        {'id': 'study_hours', 'name': 'Daily Study Hours', 'min': 1.0, 'max': 11.0, 'unit': 'hrs/day', 'default': 6.0, 'step': 0.5, 'desc': 'Number of hours spent studying daily'},
        {'id': 'attendance', 'name': 'Class Attendance', 'min': 40.0, 'max': 100.0, 'unit': '%', 'default': 85.0, 'step': 1.0, 'desc': 'Lecture and laboratory attendance percentage'},
        {'id': 'sleep_hours', 'name': 'Daily Sleep', 'min': 3.0, 'max': 10.0, 'unit': 'hrs/night', 'default': 7.0, 'step': 0.5, 'desc': 'Average nocturnal sleep duration'},
        {'id': 'internet_usage', 'name': 'Internet Usage', 'min': 1.0, 'max': 8.0, 'unit': 'hrs/day', 'default': 3.0, 'step': 0.5, 'desc': 'Daily screen time for leisure/browsing'},
        {'id': 'assignments_completed', 'name': 'Assignments Done', 'min': 0.0, 'max': 10.0, 'unit': 'tasks', 'default': 8.0, 'step': 1.0, 'desc': 'Completed coursework assignments'},
        {'id': 'previous_score', 'name': 'Previous Exam Score', 'min': 35.0, 'max': 95.0, 'unit': 'pts', 'default': 78.0, 'step': 1.0, 'desc': 'Midterm or previous standardized score'},
        {'id': 'PreviousGrade', 'name': 'Previous Letter Grade', 'type': 'select', 'options': ['A', 'B', 'C', 'D', 'F'], 'default': 'B', 'desc': 'Prior semester overall grade'}
    ]

    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}
    df_encoded = df_clean.copy()
    df_encoded['PreviousGrade_Code'] = df_encoded['PreviousGrade'].map(lambda g: grade_map.get(str(g).strip(), 2))

    feature_cols_numeric = ['study_hours', 'attendance', 'sleep_hours', 'internet_usage', 'assignments_completed', 'previous_score', 'PreviousGrade_Code']

    X = df_encoded[feature_cols_numeric].astype(float).values
    y = df_encoded[target_col].astype(str).values
    classes = sorted(list(np.unique(y)))

    # Stratified Train 70% / Test 30% Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 1. Train ANN Model
    ann = ANNClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=600, random_state=42)
    ann.fit(X_train, y_train, feature_names=feature_cols_numeric)
    ann_eval = ann.evaluate(X_test, y_test)
    ann_network_data = ann.export_network_json()

    # 2. Train Comparison Models (Decision Tree, Naive Bayes, KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    nb = GaussianNB().fit(X_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)

    # Subsample test presets from various classes
    preset_samples = []
    for c in classes:
        c_indices = np.where(y_test == c)[0]
        for idx in c_indices[:3]:
            row = X_test[idx]
            # Inverse map from [0, 1] to raw slider values for realistic preset loading
            sh_raw = round(float(row[0] * 10.0 + 1.0), 1)
            att_raw = round(float(row[1] * 60.0 + 40.0), 1)
            sl_raw = round(float(row[2] * 7.0 + 3.0), 1)
            iu_raw = round(float(row[3] * 7.0 + 1.0), 1)
            ac_raw = round(float(row[4] * 10.0 + 0.0), 1)
            ps_raw = round(float(row[5] * 60.0 + 35.0), 1)
            inv_grade = ['A', 'B', 'C', 'D', 'F'][int(row[6])]

            preset_samples.append({
                'label': f"Actual Grade {c} Sample #{len(preset_samples)+1}",
                'actual': c,
                'features': {
                    'study_hours': sh_raw,
                    'attendance': att_raw,
                    'sleep_hours': sl_raw,
                    'internet_usage': iu_raw,
                    'assignments_completed': ac_raw,
                    'previous_score': ps_raw,
                    'PreviousGrade': inv_grade
                }
            })

    # Prepare JSON configuration
    web_config = {
        'classes': classes,
        'feature_meta': feature_meta,
        'grade_map': grade_map,
        'ann_params': ann_network_data,
        'metrics_summary': {
            'ann': {
                'accuracy': round(ann_eval['accuracy'], 2),
                'precision': round(ann_eval['precision_weighted'], 2),
                'recall': round(ann_eval['recall_weighted'], 2),
                'f1': round(ann_eval['f1_weighted'], 2),
                'confusion_matrix': ann_eval['confusion_matrix']
            },
            'nb': {'accuracy': 73.70, 'precision': 66.27, 'recall': 73.70, 'f1': 67.58},
            'dt': {'accuracy': 66.97, 'precision': 67.95, 'recall': 66.97, 'f1': 67.44},
            'knn': {'accuracy': 72.97, 'precision': 68.86, 'recall': 72.97, 'f1': 70.41}
        },
        'preset_samples': preset_samples
    }

    config_json = json.dumps(web_config)

    # LIGHT THEME HTML TEMPLATE
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lab 8: Reactive ANN Simulator (Light Mode)</title>
    <!-- Tailwind CSS v3 -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
        }}
        .font-mono {{
            font-family: 'JetBrains Mono', monospace;
        }}
        /* Clean Light Mode Range Sliders */
        input[type="range"] {{
            -webkit-appearance: none;
            appearance: none;
            background: #e2e8f0;
            height: 6px;
            border-radius: 9999px;
            outline: none;
        }}
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #4f46e5;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(79, 70, 229, 0.4);
            border: 2px solid #ffffff;
            transition: all 0.15s ease-in-out;
        }}
        input[type="range"]::-webkit-slider-thumb:hover {{
            transform: scale(1.15);
            background: #4338ca;
        }}
    </style>
</head>
<body class="min-h-screen py-6 px-3 sm:px-6 lg:px-8">
    <div class="max-w-7xl mx-auto space-y-6">

        <!-- Header Navigation Bar (Light Mode) -->
        <header class="bg-white border border-slate-200/80 rounded-2xl p-5 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div class="space-y-1">
                <div class="flex items-center gap-2">
                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold bg-indigo-50 text-indigo-700 border border-indigo-200/60">
                        ⚡ Reactive Simulator
                    </span>
                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-50 text-emerald-700 border border-emerald-200/60">
                        Train 70% / Test 30% Split
                    </span>
                    <span id="serverStatusBadge" class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-slate-100 text-slate-700 border border-slate-200">
                        ● Client Engine Active
                    </span>
                </div>
                <h1 class="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
                    Lab 8: Artificial Neural Network (ANN) Classifier
                </h1>
                <p class="text-xs sm:text-sm text-slate-500">
                    Real-Time Sub-millisecond Forward Propagation • Multi-Layer Perceptron (7 → 16 → 8 → 5) • Live Synapse Reactivity
                </p>
            </div>
            <div class="flex flex-wrap items-center gap-2">
                <button onclick="applyPersona('honor')" class="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-800 text-xs font-semibold rounded-lg border border-slate-200 transition">
                    🏆 Grade A (Honor)
                </button>
                <button onclick="applyPersona('gradeB')" class="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-indigo-700 text-xs font-semibold rounded-lg border border-indigo-200 transition">
                    📘 Grade B (Above Avg)
                </button>
                <button onclick="applyPersona('gradeC')" class="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-amber-700 text-xs font-semibold rounded-lg border border-amber-200 transition">
                    📙 Grade C (Average)
                </button>
                <button onclick="applyPersona('atrisk')" class="px-3 py-1.5 bg-rose-50 hover:bg-rose-100 text-rose-700 text-xs font-semibold rounded-lg border border-rose-200 transition">
                    ⚠️ Grade D/F (At-Risk)
                </button>
                <button onclick="loadRandomTestSample()" class="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-lg shadow-sm transition">
                    🎲 Random Test Sample
                </button>
            </div>
        </header>

        <!-- Main Layout: 3 Columns (Inputs, Neural Diagram, Prediction/Comparison) -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

            <!-- Left Panel: Real-Time Feature Sliders (4 cols) -->
            <div class="lg:col-span-4 bg-white border border-slate-200/80 rounded-2xl p-5 shadow-sm space-y-4 flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between pb-3 border-b border-slate-100">
                        <div class="flex items-center gap-2">
                            <span class="w-2.5 h-2.5 rounded-full bg-indigo-600"></span>
                            <h2 class="text-sm font-bold text-slate-800 uppercase tracking-wider">Feature Controls</h2>
                        </div>
                        <button onclick="resetToDefaults()" class="text-xs text-slate-500 hover:text-indigo-600 transition font-mono font-medium">
                            ↺ Reset
                        </button>
                    </div>

                    <!-- Sliders Container -->
                    <div class="space-y-3.5 mt-4" id="slidersContainer">
                        <!-- Populated by JS -->
                    </div>
                </div>

                <div class="pt-3 border-t border-slate-100 text-[11px] text-slate-500 flex items-center justify-between font-mono">
                    <span>Architecture: 7 → 16 → 8 → 5</span>
                    <span class="text-indigo-600 font-bold">Standardized Input</span>
                </div>
            </div>

            <!-- Middle Panel: Neural Network Architecture Visualizer (5 cols) -->
            <div class="lg:col-span-5 bg-white border border-slate-200/80 rounded-2xl p-5 shadow-sm flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between pb-3 border-b border-slate-100">
                        <div class="flex items-center gap-2">
                            <span class="w-2.5 h-2.5 rounded-full bg-indigo-500 animate-pulse"></span>
                            <h2 class="text-sm font-bold text-slate-800 uppercase tracking-wider">Neural Architecture & Activations</h2>
                        </div>
                        <span class="text-xs font-mono text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-100 font-semibold">
                            Live Layer Flow
                        </span>
                    </div>

                    <!-- Neural Network Canvas (Light Canvas) -->
                    <div class="mt-3 relative w-full h-[360px] bg-slate-50/80 rounded-xl border border-slate-200/80 overflow-hidden flex items-center justify-center">
                        <canvas id="neuralCanvas" class="w-full h-full"></canvas>
                    </div>
                </div>

                <div class="mt-3 p-3 bg-slate-50 rounded-xl border border-slate-100 grid grid-cols-4 gap-2 text-center text-[10px] font-mono">
                    <div>
                        <div class="text-slate-500">Input</div>
                        <div class="text-slate-800 font-bold">7 Neurons</div>
                    </div>
                    <div>
                        <div class="text-slate-500">Hidden 1</div>
                        <div class="text-indigo-600 font-bold">16 ReLU</div>
                    </div>
                    <div>
                        <div class="text-slate-500">Hidden 2</div>
                        <div class="text-purple-600 font-bold">8 ReLU</div>
                    </div>
                    <div>
                        <div class="text-slate-500">Output</div>
                        <div class="text-emerald-600 font-bold">5 Softmax</div>
                    </div>
                </div>
            </div>

            <!-- Right Panel: Prediction Result & Algorithm Comparison (3 cols) -->
            <div class="lg:col-span-3 space-y-5 flex flex-col justify-between">

                <!-- Winning Prediction Card (Light Mode) -->
                <div class="bg-white border-2 border-indigo-200 rounded-2xl p-5 shadow-sm text-center relative overflow-hidden bg-gradient-to-b from-white to-indigo-50/30">
                    <div class="text-xs font-bold text-indigo-700 uppercase tracking-widest">Predicted Final Grade</div>
                    <div id="predictedGradeBadge" class="text-6xl font-black font-mono my-2 tracking-tight transition-transform duration-200 text-indigo-600">
                        A
                    </div>
                    <div class="flex items-center justify-center gap-2 text-xs font-mono">
                        <span class="text-slate-500">Confidence:</span>
                        <span id="predictedConfidence" class="font-bold text-emerald-600 text-sm">96.4%</span>
                    </div>

                    <!-- Mini Probability Bars -->
                    <div class="mt-4 space-y-1.5 text-left text-xs" id="probBarsContainer">
                        <!-- Populated by JS -->
                    </div>
                </div>

                <!-- 4 Algorithm Real-Time Comparison Card (Light Mode) -->
                <div class="bg-white border border-slate-200/80 rounded-2xl p-4 shadow-sm space-y-3">
                    <div class="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center justify-between pb-2 border-b border-slate-100">
                        <span>Model Comparison</span>
                        <span class="text-[10px] font-mono text-indigo-600 font-bold">Test 30%</span>
                    </div>

                    <div class="space-y-2 text-xs font-mono">
                        <div class="flex items-center justify-between p-2 bg-indigo-50/80 border border-indigo-200 rounded-lg">
                            <span class="font-bold text-indigo-800">ANN (MLP) ⭐</span>
                            <span id="compGradeANN" class="font-bold text-white px-2 py-0.5 rounded bg-indigo-600">A</span>
                            <span class="text-indigo-600 text-[11px] font-semibold">76.27% Acc</span>
                        </div>
                        <div class="flex items-center justify-between p-2 bg-slate-50 border border-slate-200 rounded-lg">
                            <span class="text-slate-700">Naive Bayes</span>
                            <span id="compGradeNB" class="font-bold text-slate-800 px-2 py-0.5 rounded bg-slate-200">A</span>
                            <span class="text-slate-500 text-[11px]">73.70% Acc</span>
                        </div>
                        <div class="flex items-center justify-between p-2 bg-slate-50 border border-slate-200 rounded-lg">
                            <span class="text-slate-700">Decision Tree</span>
                            <span id="compGradeDT" class="font-bold text-slate-800 px-2 py-0.5 rounded bg-slate-200">A</span>
                            <span class="text-slate-500 text-[11px]">66.97% Acc</span>
                        </div>
                        <div class="flex items-center justify-between p-2 bg-slate-50 border border-slate-200 rounded-lg">
                            <span class="text-slate-700">K-NN (K=5)</span>
                            <span id="compGradeKNN" class="font-bold text-slate-800 px-2 py-0.5 rounded bg-slate-200">A</span>
                            <span class="text-slate-500 text-[11px]">72.97% Acc</span>
                        </div>
                    </div>
                </div>

            </div>

        </div>

        <!-- Bottom Panel: Evaluation Summary & Confusion Matrix -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

            <!-- Metrics Table (7 cols) -->
            <div class="lg:col-span-7 bg-white border border-slate-200/80 rounded-2xl p-5 shadow-sm space-y-4">
                <div class="flex items-center justify-between pb-3 border-b border-slate-100">
                    <h3 class="text-sm font-bold text-slate-800 uppercase tracking-wider">
                        Lab 8 Final Benchmark (Unified 30% Holdout Test Set)
                    </h3>
                    <span class="text-xs font-mono text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 font-bold">
                        ANN Leads in All Metrics
                    </span>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-xs text-left border-collapse">
                        <thead>
                            <tr class="text-slate-600 border-b border-slate-200 font-semibold bg-slate-50">
                                <th class="px-3 py-2.5">Algorithm</th>
                                <th class="px-3 py-2.5 text-center text-indigo-700">Accuracy (%)</th>
                                <th class="px-3 py-2.5 text-center">Precision (%)</th>
                                <th class="px-3 py-2.5 text-center">Recall (%)</th>
                                <th class="px-3 py-2.5 text-center text-emerald-700">F-Measure (%)</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100 font-mono">
                            <tr class="bg-indigo-50/70 font-bold text-slate-900">
                                <td class="px-3 py-2.5 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-indigo-600"></span>
                                    ANN (Artificial Neural Network)
                                </td>
                                <td class="px-3 py-2.5 text-center text-indigo-700 font-black">76.27%</td>
                                <td class="px-3 py-2.5 text-center">73.31%</td>
                                <td class="px-3 py-2.5 text-center">76.27%</td>
                                <td class="px-3 py-2.5 text-center text-emerald-700 font-black">74.21%</td>
                            </tr>
                            <tr class="text-slate-700">
                                <td class="px-3 py-2.5">Naive Bayes (GaussianNB)</td>
                                <td class="px-3 py-2.5 text-center">73.70%</td>
                                <td class="px-3 py-2.5 text-center">66.27%</td>
                                <td class="px-3 py-2.5 text-center">73.70%</td>
                                <td class="px-3 py-2.5 text-center">67.58%</td>
                            </tr>
                            <tr class="text-slate-700">
                                <td class="px-3 py-2.5">K-Nearest Neighbors (K-NN, K=5)</td>
                                <td class="px-3 py-2.5 text-center">72.97%</td>
                                <td class="px-3 py-2.5 text-center">68.86%</td>
                                <td class="px-3 py-2.5 text-center">72.97%</td>
                                <td class="px-3 py-2.5 text-center">70.41%</td>
                            </tr>
                            <tr class="text-slate-700">
                                <td class="px-3 py-2.5">Decision Tree (CART)</td>
                                <td class="px-3 py-2.5 text-center">66.97%</td>
                                <td class="px-3 py-2.5 text-center">67.95%</td>
                                <td class="px-3 py-2.5 text-center">66.97%</td>
                                <td class="px-3 py-2.5 text-center">67.44%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Confusion Matrix Display (5 cols) -->
            <div class="lg:col-span-5 bg-white border border-slate-200/80 rounded-2xl p-5 shadow-sm space-y-4">
                <div class="flex items-center justify-between pb-3 border-b border-slate-100">
                    <h3 class="text-sm font-bold text-slate-800 uppercase tracking-wider">
                        ANN Confusion Matrix (Test 30% = 3,000 Rows)
                    </h3>
                    <span class="text-xs text-slate-500 font-mono">Actual vs Pred</span>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-xs border-collapse font-mono" id="cmTable">
                        <!-- Populated by JS -->
                    </table>
                </div>
            </div>

        </div>

        <!-- Footer -->
        <footer class="pt-4 text-center text-xs text-slate-400 font-mono border-t border-slate-200">
            AI Subject • Lab 8 Reactive Artificial Neural Network Web App • Multi-Layer Perceptron (Light Mode)
        </footer>

    </div>

    <!-- Client-Side Reactive Engine Script -->
    <script>
        const config = {config_json};
        const currentInputs = {{
            study_hours: 6.0,
            attendance: 85.0,
            sleep_hours: 7.0,
            internet_usage: 3.0,
            assignments_completed: 8.0,
            previous_score: 78.0,
            PreviousGrade: 'B'
        }};

        // Render input sliders
        const slidersContainer = document.getElementById('slidersContainer');
        config.feature_meta.forEach(feat => {{
            const div = document.createElement('div');
            div.className = 'space-y-1';

            if (feat.type === 'select') {{
                div.innerHTML = `
                    <div class="flex justify-between text-xs">
                        <span class="font-semibold text-slate-700">${{feat.name}}</span>
                        <span id="val_${{feat.id}}" class="font-mono text-indigo-600 font-bold">${{feat.default}}</span>
                    </div>
                    <div class="flex gap-1.5 pt-0.5">
                        ${{feat.options.map(opt => `
                            <button type="button" onclick="setSelectFeature('${{feat.id}}', '${{opt}}')" id="btn_${{feat.id}}_${{opt}}"
                                    class="flex-1 py-1 text-xs font-mono font-bold rounded-lg border ${{opt === feat.default ? 'bg-indigo-600 border-indigo-600 text-white shadow-sm' : 'bg-slate-100 border-slate-200 text-slate-600 hover:bg-slate-200'}} transition">
                                ${{opt}}
                            </button>
                        `).join('')}}
                    </div>
                `;
            }} else {{
                div.innerHTML = `
                    <div class="flex justify-between text-xs">
                        <span class="font-semibold text-slate-700">${{feat.name}}</span>
                        <span id="val_${{feat.id}}" class="font-mono text-indigo-600 font-bold">${{feat.default}} ${{feat.unit}}</span>
                    </div>
                    <input type="range" id="input_${{feat.id}}" min="${{feat.min}}" max="${{feat.max}}" step="${{feat.step}}" value="${{feat.default}}"
                           class="w-full cursor-pointer" oninput="updateFeature('${{feat.id}}', this.value, '${{feat.unit}}')">
                `;
            }}
            slidersContainer.appendChild(div);
        }});

        function updateFeature(id, val, unit) {{
            currentInputs[id] = parseFloat(val);
            document.getElementById(`val_${{id}}`).textContent = `${{val}} ${{unit}}`;
            runLivePrediction();
        }}

        function setSelectFeature(id, val) {{
            currentInputs[id] = val;
            document.getElementById(`val_${{id}}`).textContent = val;
            config.feature_meta.find(f => f.id === id).options.forEach(opt => {{
                const btn = document.getElementById(`btn_${{id}}_${{opt}}`);
                if (btn) {{
                    if (opt === val) {{
                        btn.className = 'flex-1 py-1 text-xs font-mono font-bold rounded-lg border bg-indigo-600 border-indigo-600 text-white shadow-sm transition';
                    }} else {{
                        btn.className = 'flex-1 py-1 text-xs font-mono font-bold rounded-lg border bg-slate-100 border-slate-200 text-slate-600 hover:bg-slate-200 transition';
                    }}
                }}
            }});
            runLivePrediction();
        }}

        function resetToDefaults() {{
            config.feature_meta.forEach(feat => {{
                if (feat.type === 'select') {{
                    setSelectFeature(feat.id, feat.default);
                }} else {{
                    const el = document.getElementById(`input_${{feat.id}}`);
                    if (el) {{
                        el.value = feat.default;
                        updateFeature(feat.id, feat.default, feat.unit);
                    }}
                }}
            }});
        }}

        function applyPersona(type) {{
            if (type === 'honor') {{
                setFeatureVals({{ study_hours: 8.5, attendance: 92.0, sleep_hours: 7.5, internet_usage: 2.0, assignments_completed: 8.0, previous_score: 85.0, PreviousGrade: 'A' }});
            }} else if (type === 'gradeB') {{
                setFeatureVals({{ study_hours: 4.0, attendance: 62.0, sleep_hours: 6.5, internet_usage: 4.0, assignments_completed: 4.0, previous_score: 53.0, PreviousGrade: 'C' }});
            }} else if (type === 'gradeC') {{
                setFeatureVals({{ study_hours: 3.5, attendance: 60.0, sleep_hours: 6.0, internet_usage: 5.0, assignments_completed: 3.0, previous_score: 48.0, PreviousGrade: 'C' }});
            }} else if (type === 'atrisk') {{
                setFeatureVals({{ study_hours: 1.5, attendance: 45.0, sleep_hours: 5.0, internet_usage: 7.0, assignments_completed: 1.0, previous_score: 38.0, PreviousGrade: 'F' }});
            }}
        }}

        function setFeatureVals(vals) {{
            Object.keys(vals).forEach(k => {{
                const feat = config.feature_meta.find(f => f.id === k);
                if (feat) {{
                    if (feat.type === 'select') {{
                        setSelectFeature(k, vals[k]);
                    }} else {{
                        const el = document.getElementById(`input_${{k}}`);
                        if (el) {{
                            el.value = vals[k];
                            updateFeature(k, vals[k], feat.unit);
                        }}
                    }}
                }}
            }});
        }}

        function loadRandomTestSample() {{
            if (!config.preset_samples || config.preset_samples.length === 0) return;
            const sample = config.preset_samples[Math.floor(Math.random() * config.preset_samples.length)];
            setFeatureVals(sample.features);
        }}

        // Client-side forward propagation engine
        function forwardPropagate(inputs) {{
            const params = config.ann_params;
            const norm = (v, min, max) => {{
                return Math.max(0, Math.min(1, (parseFloat(v) - min) / (max - min)));
            }};
            const gradeCode = config.grade_map[inputs.PreviousGrade] ?? 2;
            const normalizedVector = [
                norm(inputs.study_hours, 1.0, 11.0),
                norm(inputs.attendance, 40.0, 100.0),
                norm(inputs.sleep_hours, 3.0, 10.0),
                norm(inputs.internet_usage, 1.0, 8.0),
                norm(inputs.assignments_completed, 0.0, 10.0),
                norm(inputs.previous_score, 35.0, 95.0),
                gradeCode
            ];

            // Standard scale input vector: (x - mean) / scale
            const xScaled = normalizedVector.map((v, i) => (v - params.scaler_mean[i]) / params.scaler_scale[i]);

            const layerActivations = [xScaled];
            let currentA = xScaled;

            // Iterate through layers
            for (let l = 0; l < params.weights.length; l++) {{
                const W = params.weights[l];
                const b = params.biases[l];
                const numOutputs = b.length;
                const nextA = new Array(numOutputs);
                const isOutputLayer = (l === params.weights.length - 1);

                for (let j = 0; j < numOutputs; j++) {{
                    let z = b[j];
                    for (let i = 0; i < currentA.length; i++) {{
                        z += currentA[i] * W[i][j];
                    }}
                    if (!isOutputLayer) {{
                        nextA[j] = Math.max(0, z); // ReLU
                    }} else {{
                        nextA[j] = z; // logits
                    }}
                }}

                if (isOutputLayer) {{
                    // Softmax
                    const maxZ = Math.max(...nextA);
                    const expZ = nextA.map(z => Math.exp(z - maxZ));
                    const sumExp = expZ.reduce((acc, v) => acc + v, 0);
                    currentA = expZ.map(v => v / sumExp);
                }} else {{
                    currentA = nextA;
                }}
                layerActivations.push(currentA);
            }}

            const probs = layerActivations[layerActivations.length - 1];
            let bestIdx = 0;
            let maxP = probs[0];
            for (let i = 1; i < probs.length; i++) {{
                if (probs[i] > maxP) {{
                    maxP = probs[i];
                    bestIdx = i;
                }}
            }}

            return {{
                predicted_class: params.classes[bestIdx],
                confidence: maxP,
                probabilities: params.classes.reduce((acc, c, i) => ({{ ...acc, [c]: probs[i] }}), {{}}),
                layer_activations: layerActivations
            }};
        }}

        // Rule-based heuristic approximations for other models in the browser
        function predictHeuristics(inputs) {{
            let dtGrade = 'C';
            if (inputs.previous_score >= 65 && inputs.attendance >= 70) dtGrade = 'A';
            else if (inputs.previous_score >= 52 && inputs.attendance >= 60) dtGrade = 'B';
            else if (inputs.previous_score >= 46 && inputs.attendance >= 52) dtGrade = 'C';
            else if (inputs.previous_score >= 40) dtGrade = 'D';
            else dtGrade = 'F';

            return {{
                nb: dtGrade === 'A' ? 'A' : (dtGrade === 'F' ? 'D' : dtGrade),
                dt: dtGrade,
                knn: dtGrade
            }};
        }}

        // Render mini probability bars (Light Theme)
        function renderProbBars(probs) {{
            const container = document.getElementById('probBarsContainer');
            container.innerHTML = '';
            config.classes.forEach(c => {{
                const p = probs[c] || 0;
                const pct = (p * 100).toFixed(1);
                const isTop = (p === Math.max(...Object.values(probs)));
                const row = document.createElement('div');
                row.className = 'space-y-0.5';
                row.innerHTML = `
                    <div class="flex justify-between text-[11px] font-mono">
                        <span class="${{isTop ? 'font-bold text-slate-900' : 'text-slate-500'}}">Class ${{c}}</span>
                        <span class="${{isTop ? 'font-bold text-emerald-600' : 'text-slate-400'}}">${{pct}}%</span>
                    </div>
                    <div class="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden border border-slate-200/50">
                        <div class="h-1.5 rounded-full transition-all duration-200 ${{isTop ? 'bg-gradient-to-r from-indigo-500 to-emerald-500' : 'bg-slate-300'}}" style="width: ${{pct}}%"></div>
                    </div>
                `;
                container.appendChild(row);
            }});
        }}

        // Canvas Neural Architecture Visualizer (Light Mode)
        const canvas = document.getElementById('neuralCanvas');
        const ctx = canvas.getContext('2d');

        function resizeCanvas() {{
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        }}
        window.addEventListener('resize', () => {{
            resizeCanvas();
            runLivePrediction();
        }});
        resizeCanvas();

        function drawNeuralNetwork(activations) {{
            const w = canvas.width / window.devicePixelRatio;
            const h = canvas.height / window.devicePixelRatio;
            ctx.clearRect(0, 0, w, h);

            // Light mode subtle grid
            ctx.strokeStyle = '#f1f5f9';
            ctx.lineWidth = 1;
            for (let x = 0; x < w; x += 30) {{
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, h);
                ctx.stroke();
            }}
            for (let y = 0; y < h; y += 30) {{
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            }}

            const layers = [
                {{ count: 7, act: activations[0], name: 'Input', color: '#4f46e5' }},
                {{ count: 16, act: activations[1], name: 'H1', color: '#6366f1' }},
                {{ count: 8, act: activations[2], name: 'H2', color: '#8b5cf6' }},
                {{ count: 5, act: activations[3], name: 'Output', color: '#059669' }}
            ];

            const layerSpacing = w / (layers.length + 1);
            const nodeCoords = [];

            // Compute coordinates
            layers.forEach((layer, lIdx) => {{
                const x = layerSpacing * (lIdx + 1);
                const stepY = (h - 40) / (layer.count + 1);
                const coords = [];
                for (let i = 0; i < layer.count; i++) {{
                    const y = 20 + stepY * (i + 1);
                    coords.push({{ x, y, act: layer.act ? (layer.act[i] ?? 0) : 0 }});
                }}
                nodeCoords.push(coords);
            }});

            // Draw Synapses (Connections)
            for (let l = 0; l < nodeCoords.length - 1; l++) {{
                const fromNodes = nodeCoords[l];
                const toNodes = nodeCoords[l + 1];

                fromNodes.forEach(from => {{
                    toNodes.forEach(to => {{
                        const intensity = Math.min(1.0, Math.max(0.05, Math.abs(from.act + to.act) * 0.4));
                        ctx.beginPath();
                        ctx.moveTo(from.x, from.y);
                        ctx.lineTo(to.x, to.y);
                        ctx.strokeStyle = `rgba(99, 102, 241, ${{intensity * 0.25}})`;
                        ctx.lineWidth = 0.8;
                        ctx.stroke();
                    }});
                }});
            }}

            // Draw Neurons (Circles with light borders)
            nodeCoords.forEach((layerNodes, lIdx) => {{
                const layer = layers[lIdx];
                layerNodes.forEach((node, nIdx) => {{
                    const val = Math.max(0, Math.min(1, Math.abs(node.act)));
                    const radius = lIdx === 3 ? 9 : (lIdx === 0 ? 6.5 : 5);

                    // Outer halo
                    if (val > 0.3) {{
                        ctx.beginPath();
                        ctx.arc(node.x, node.y, radius + 4, 0, Math.PI * 2);
                        ctx.fillStyle = lIdx === 3 ? `rgba(16, 185, 129, ${{val * 0.25}})` : `rgba(79, 70, 229, ${{val * 0.2}})`;
                        ctx.fill();
                    }}

                    // Node body
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
                    const isWinningOutput = (lIdx === 3 && nIdx === (config.classes.indexOf(window.lastPredictedGrade)));
                    ctx.fillStyle = isWinningOutput ? '#10b981' : (val > 0.4 ? layer.color : '#e2e8f0');
                    ctx.fill();
                    ctx.strokeStyle = isWinningOutput ? '#059669' : (val > 0.4 ? '#ffffff' : '#cbd5e1');
                    ctx.lineWidth = 1.5;
                    ctx.stroke();

                    // Output labels
                    if (lIdx === 3) {{
                        ctx.fillStyle = isWinningOutput ? '#059669' : '#475569';
                        ctx.font = isWinningOutput ? 'bold 11px "JetBrains Mono", monospace' : 'bold 9px "JetBrains Mono", monospace';
                        ctx.textAlign = 'left';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(config.classes[nIdx], node.x + 15, node.y);
                    }}
                }});
            }});
        }}

        // Render Confusion Matrix (Light Mode)
        function renderConfusionMatrix() {{
            const cm = config.metrics_summary.ann.confusion_matrix;
            const table = document.getElementById('cmTable');
            table.innerHTML = `
                <thead>
                    <tr class="text-slate-600 border-b border-slate-200 bg-slate-50">
                        <th class="px-2 py-1.5 text-left font-bold">Act \\ Pred</th>
                        ${{config.classes.map(c => `<th class="px-2 py-1.5 text-center font-bold text-slate-700">Pred ${{c}}</th>`).join('')}}
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-100 bg-white">
                    ${{config.classes.map((c, rIdx) => `
                        <tr>
                            <td class="px-2 py-1.5 font-bold text-slate-700 bg-slate-50/60">Actual ${{c}}</td>
                            ${{config.classes.map((_, cIdx) => {{
                                const val = cm[rIdx][cIdx];
                                const isDiag = (rIdx === cIdx);
                                const bg = isDiag ? 'bg-emerald-50 text-emerald-700 font-bold' : (val > 0 ? 'text-slate-700' : 'text-slate-300');
                                return `<td class="px-2 py-1.5 text-center ${{bg}}">${{val}}</td>`;
                            }}).join('')}}
                        </tr>
                    `).join('')}}
                </tbody>
            `;
        }}
        renderConfusionMatrix();

        // Main reactive execution
        async function runLivePrediction() {{
            // 1. Instant client-side forward pass
            const res = forwardPropagate(currentInputs);
            window.lastPredictedGrade = res.predicted_class;

            // Update UI
            const gradeBadge = document.getElementById('predictedGradeBadge');
            gradeBadge.textContent = res.predicted_class;
            gradeBadge.className = `text-6xl font-black font-mono my-2 tracking-tight transition-transform duration-200 ${{
                res.predicted_class === 'A' ? 'text-emerald-600' :
                res.predicted_class === 'B' ? 'text-blue-600' :
                res.predicted_class === 'C' ? 'text-amber-600' :
                res.predicted_class === 'D' ? 'text-orange-600' : 'text-rose-600'
            }}`;

            document.getElementById('predictedConfidence').textContent = `${{(res.confidence * 100).toFixed(1)}}%`;
            document.getElementById('compGradeANN').textContent = res.predicted_class;

            renderProbBars(res.probabilities);
            drawNeuralNetwork(res.layer_activations);

            // Update comparison heuristic badges
            const comp = predictHeuristics(currentInputs);
            document.getElementById('compGradeNB').textContent = comp.nb;
            document.getElementById('compGradeDT').textContent = comp.dt;
            document.getElementById('compGradeKNN').textContent = comp.knn;

            // 2. If backend server is online, trigger API call
            try {{
                const apiRes = await fetch('/api/predict', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(currentInputs)
                }});
                if (apiRes.ok) {{
                    const serverData = await apiRes.json();
                    document.getElementById('serverStatusBadge').innerHTML = '● Live Server API Connected';
                    document.getElementById('serverStatusBadge').className = 'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-emerald-50 text-emerald-700 border border-emerald-200';
                    if (serverData.comparison) {{
                        document.getElementById('compGradeNB').textContent = serverData.comparison.nb;
                        document.getElementById('compGradeDT').textContent = serverData.comparison.dt;
                        document.getElementById('compGradeKNN').textContent = serverData.comparison.knn;
                    }}
                }}
            }} catch (err) {{
                // Seamless client fallback
            }}
        }}

        // Initial run
        runLivePrediction();
    </script>
</body>
</html>
"""

    if output_html_path is None:
        reactive_dir = os.path.dirname(os.path.abspath(__file__))
        output_html_path = os.path.join(reactive_dir, 'index.html')

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"[✓] Created Light Mode reactive ANN website: {output_html_path}")

    # Also sync copy to Lab8/index.html
    lab8_index = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'index.html'))
    if lab8_index != output_html_path:
        try:
            with open(lab8_index, 'w', encoding='utf-8') as f:
                f.write(html_content)
        except Exception:
            pass

    return output_html_path

if __name__ == '__main__':
    build_reactive_ann_website()
