import os
import json
import html
from datetime import datetime

def generate_lab5_html_report(dataset_info, all_model_stats, comparison_data, best_models, classes, output_path=None):
    """
    Generate a clean, simplified, responsive HTML report comparing the 3 models
    across 5-Fold Cross Validation.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Summary comparison table rows
    comparison_table_rows = []
    for m in comparison_data:
        comparison_table_rows.append(f"""
        <tr class="hover:bg-slate-50 transition-colors">
            <td class="px-4 py-3 font-semibold text-slate-800">{html.escape(m['Model'])}</td>
            <td class="px-4 py-3 text-center font-mono font-bold text-indigo-700 bg-indigo-50/40">{m['Accuracy (%)']}</td>
            <td class="px-4 py-3 text-center font-mono text-slate-700">{m['Precision (%)']}</td>
            <td class="px-4 py-3 text-center font-mono text-slate-700">{m['Recall (%)']}</td>
            <td class="px-4 py-3 text-center font-mono font-bold text-emerald-700 bg-emerald-50/40">{m['F-Measure (%)']}</td>
        </tr>
        """)

    # 2. Per-model fold sections (Accordion / Tab style)
    model_sections_html = []
    for model_idx, model_data in enumerate(all_model_stats, start=1):
        m_name = model_data['Model']
        acc_avg = model_data['Accuracy_mean']
        f1_avg = model_data['F-Measure_mean']

        # 5 Folds details table
        fold_rows = []
        for f in model_data['fold_details']:
            f_num = f['fold_num']
            fold_rows.append(f"""
            <tr class="hover:bg-slate-50 border-b border-slate-100">
                <td class="px-3 py-2 font-medium text-slate-700">Fold {f_num}</td>
                <td class="px-3 py-2 text-center font-mono font-semibold text-indigo-600">{f['accuracy']:.2f}%</td>
                <td class="px-3 py-2 text-center font-mono text-slate-600">{f['prec_weighted']:.2f}%</td>
                <td class="px-3 py-2 text-center font-mono text-slate-600">{f['rec_weighted']:.2f}%</td>
                <td class="px-3 py-2 text-center font-mono font-semibold text-emerald-600">{f['f1_weighted']:.2f}%</td>
            </tr>
            """)

        # Add Mean and Std rows
        fold_rows.append(f"""
        <tr class="bg-slate-100/80 font-bold border-t-2 border-slate-200">
            <td class="px-3 py-2 text-slate-800">Mean (Average)</td>
            <td class="px-3 py-2 text-center font-mono text-indigo-700">{model_data['Accuracy_mean']:.2f}%</td>
            <td class="px-3 py-2 text-center font-mono text-slate-800">{model_data['Precision_mean']:.2f}%</td>
            <td class="px-3 py-2 text-center font-mono text-slate-800">{model_data['Recall_mean']:.2f}%</td>
            <td class="px-3 py-2 text-center font-mono text-emerald-700">{model_data['F-Measure_mean']:.2f}%</td>
        </tr>
        <tr class="bg-slate-50 text-xs text-slate-500 font-medium">
            <td class="px-3 py-1.5 text-slate-500">Std Deviation (±)</td>
            <td class="px-3 py-1.5 text-center font-mono">±{model_data['Accuracy_std']:.2f}%</td>
            <td class="px-3 py-1.5 text-center font-mono">±{model_data['Precision_std']:.2f}%</td>
            <td class="px-3 py-1.5 text-center font-mono">±{model_data['Recall_std']:.2f}%</td>
            <td class="px-3 py-1.5 text-center font-mono">±{model_data['F-Measure_std']:.2f}%</td>
        </tr>
        """)

        # 5 Actual Tables (Confusion Matrices)
        cm_cards = []
        for f in model_data['fold_details']:
            f_num = f['fold_num']
            cm = f['confusion_matrix']
            cm_max = max(max(row) for row in cm) if cm else 1

            cm_table_rows = []
            for r_idx, c_name in enumerate(classes):
                row_sum = sum(cm[r_idx])
                row_cells = [f'<td class="px-2 py-1 text-left font-medium text-slate-600 bg-slate-50">Actual {c_name}</td>']
                for c_idx, pred_name in enumerate(classes):
                    val = cm[r_idx][c_idx]
                    is_diag = (r_idx == c_idx)
                    bg_color = "bg-emerald-100 font-bold text-emerald-800" if is_diag and val > 0 else ("bg-rose-50 text-rose-700 font-medium" if val > 0 else "text-slate-400")
                    row_cells.append(f'<td class="px-2 py-1 text-center font-mono text-xs {bg_color}">{val}</td>')
                row_cells.append(f'<td class="px-2 py-1 text-center font-mono text-xs font-semibold text-slate-500">{row_sum}</td>')
                cm_table_rows.append(f"<tr>{''.join(row_cells)}</tr>")

            cm_cards.append(f"""
            <div class="bg-white border border-slate-200 rounded-lg p-3 shadow-sm">
                <div class="flex justify-between items-center mb-2 pb-1 border-b border-slate-100 text-xs">
                    <span class="font-bold text-slate-700">Fold {f_num} Actual Table</span>
                    <span class="font-mono text-indigo-600 font-semibold">Acc: {f['accuracy']:.1f}%</span>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-[11px] border-collapse">
                        <thead>
                            <tr class="text-slate-500 border-b border-slate-200">
                                <th class="px-1 py-1 text-left font-semibold">Act \\ Pred</th>
                                {''.join([f'<th class="px-1 py-1 text-center font-semibold text-slate-700">{c}</th>' for c in classes])}
                                <th class="px-1 py-1 text-center font-semibold text-slate-400">Total</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100">
                            {''.join(cm_table_rows)}
                        </tbody>
                    </table>
                </div>
            </div>
            """)

        model_sections_html.append(f"""
        <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-100 gap-2">
                <div class="flex items-center gap-3">
                    <span class="w-8 h-8 rounded-lg bg-indigo-600 text-white font-bold flex items-center justify-center text-sm shadow-sm">
                        {model_idx}
                    </span>
                    <div>
                        <h3 class="text-lg font-bold text-slate-800">{html.escape(m_name)}</h3>
                        <p class="text-xs text-slate-500">20% Stratified Validation per fold (2,000 samples / fold)</p>
                    </div>
                </div>
                <div class="flex items-center gap-3 text-xs font-mono">
                    <span class="px-2.5 py-1 rounded bg-indigo-50 text-indigo-700 font-bold border border-indigo-100">
                        Accuracy: {acc_avg:.2f}%
                    </span>
                    <span class="px-2.5 py-1 rounded bg-emerald-50 text-emerald-700 font-bold border border-emerald-100">
                        F-Measure: {f1_avg:.2f}%
                    </span>
                </div>
            </div>

            <!-- 5 Folds Summary Table -->
            <div class="overflow-x-auto">
                <table class="w-full text-xs text-left border-collapse">
                    <thead>
                        <tr class="bg-slate-50 text-slate-600 border-b border-slate-200 font-semibold">
                            <th class="px-3 py-2">Validation Split</th>
                            <th class="px-3 py-2 text-center text-indigo-700">Accuracy</th>
                            <th class="px-3 py-2 text-center">Precision (Weighted)</th>
                            <th class="px-3 py-2 text-center">Recall (Weighted)</th>
                            <th class="px-3 py-2 text-center text-emerald-700">F-Measure (Weighted)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(fold_rows)}
                    </tbody>
                </table>
            </div>

            <!-- 5 Confusion Matrices in a compact grid -->
            <div>
                <h4 class="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                    5 Folds Actual Tables (Confusion Matrices)
                </h4>
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
                    {''.join(cm_cards)}
                </div>
            </div>
        </div>
        """)

    # 3. Clean HTML Document template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lab 5: Cross-Validation & Model Comparison Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
        }}
        .font-mono {{
            font-family: 'JetBrains Mono', monospace;
        }}
    </style>
</head>
<body class="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
    <div class="max-w-6xl mx-auto space-y-6">

        <!-- Header -->
        <div class="bg-white border border-slate-200 rounded-xl p-6 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
                <span class="text-xs font-bold tracking-wider text-indigo-600 uppercase">Machine Learning Evaluation</span>
                <h1 class="text-2xl font-bold text-slate-900 mt-0.5">Lab 5: Model Evaluation & Comparison</h1>
                <p class="text-xs text-slate-500 mt-1">
                    5-Fold Cross Validation (20% Holdout per fold) on unified normalized dataset (10,000 samples)
                </p>
            </div>
            <div class="flex flex-wrap items-center gap-2 text-xs font-mono text-slate-600">
                <span class="px-2.5 py-1 bg-slate-100 rounded border border-slate-200">Dataset: {html.escape(dataset_info['dataset_name'])}</span>
                <span class="px-2.5 py-1 bg-slate-100 rounded border border-slate-200">Rows: {dataset_info['total_samples']:,}</span>
                <span class="px-2.5 py-1 bg-indigo-50 text-indigo-700 rounded border border-indigo-200 font-semibold">{timestamp}</span>
            </div>
        </div>

        <!-- KPI Cards -->
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div class="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                <span class="text-xs font-semibold text-slate-500 uppercase">Highest Accuracy</span>
                <div class="text-2xl font-bold text-indigo-600 font-mono mt-1">{best_models['acc']['Accuracy_mean']:.2f}%</div>
                <div class="text-xs text-slate-600 font-medium mt-0.5">{html.escape(best_models['acc']['Model'])}</div>
            </div>
            <div class="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                <span class="text-xs font-semibold text-slate-500 uppercase">Highest F-Measure</span>
                <div class="text-2xl font-bold text-emerald-600 font-mono mt-1">{best_models['f1']['F-Measure_mean']:.2f}%</div>
                <div class="text-xs text-slate-600 font-medium mt-0.5">{html.escape(best_models['f1']['Model'])}</div>
            </div>
            <div class="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                <span class="text-xs font-semibold text-slate-500 uppercase">Highest Precision</span>
                <div class="text-2xl font-bold text-blue-600 font-mono mt-1">{best_models['prec']['Precision_mean']:.2f}%</div>
                <div class="text-xs text-slate-600 font-medium mt-0.5">{html.escape(best_models['prec']['Model'])}</div>
            </div>
        </div>

        <!-- Overall Comparison Table & Chart -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <!-- Table (7 cols) -->
            <div class="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col justify-between">
                <div>
                    <h2 class="text-base font-bold text-slate-800 pb-3 border-b border-slate-100">
                        Final Model Comparison (Average ± Std across 5 Folds)
                    </h2>
                    <div class="overflow-x-auto mt-3">
                        <table class="w-full text-xs text-left border-collapse">
                            <thead>
                                <tr class="bg-slate-50 text-slate-600 border-b border-slate-200 font-semibold">
                                    <th class="px-4 py-2.5">Algorithm</th>
                                    <th class="px-4 py-2.5 text-center text-indigo-700">Accuracy (%)</th>
                                    <th class="px-4 py-2.5 text-center">Precision (%)</th>
                                    <th class="px-4 py-2.5 text-center">Recall (%)</th>
                                    <th class="px-4 py-2.5 text-center text-emerald-700">F-Measure (%)</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-100">
                                {''.join(comparison_table_rows)}
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="text-[11px] text-slate-500 pt-3 border-t border-slate-100 mt-4">
                    * Metrics calculated with weighted averaging across target classes: {', '.join(classes)}
                </div>
            </div>

            <!-- Chart (5 cols) -->
            <div class="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col justify-between">
                <h2 class="text-base font-bold text-slate-800 pb-2 border-b border-slate-100">
                    Accuracy & F-Measure Visual Comparison
                </h2>
                <div class="h-56 mt-2 relative">
                    <canvas id="comparisonChart"></canvas>
                </div>
                <div class="text-center text-[11px] text-slate-400 font-mono pt-2 border-t border-slate-100">
                    5-Fold Cross Validation Mean Results
                </div>
            </div>
        </div>

        <!-- Individual Model Sections (Actual Tables + Fold Metrics) -->
        <div class="space-y-6">
            <h2 class="text-lg font-bold text-slate-900">
                Detailed 5-Fold Evaluation by Algorithm
            </h2>
            {''.join(model_sections_html)}
        </div>

        <!-- Footer -->
        <footer class="pt-6 pb-2 text-center text-xs text-slate-400 font-mono border-t border-slate-200">
            AI Subject • Lab 5 Model Evaluation • Generated: {timestamp}
        </footer>

    </div>

    <!-- Chart.js Script -->
    <script>
        const ctx = document.getElementById('comparisonChart').getContext('2d');
        const models = {json.dumps([m['Model'] for m in all_model_stats])};
        const accs = {json.dumps([m['Accuracy_mean'] for m in all_model_stats])};
        const f1s = {json.dumps([m['F-Measure_mean'] for m in all_model_stats])};

        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: models,
                datasets: [
                    {{
                        label: 'Accuracy (%)',
                        data: accs,
                        backgroundColor: '#6366f1',
                        borderRadius: 4
                    }},
                    {{
                        label: 'F-Measure (%)',
                        data: f1s,
                        backgroundColor: '#10b981',
                        borderRadius: 4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'top',
                        labels: {{ boxWidth: 12, font: {{ size: 11 }} }}
                    }}
                }},
                scales: {{
                    y: {{
                        min: 60,
                        max: 80,
                        ticks: {{ callback: v => v + '%' }}
                    }},
                    x: {{
                        ticks: {{ font: {{ size: 10, weight: 'bold' }} }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    target_file = output_path
    if not target_file:
        target_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lab5_report.html')

    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n[✓] Saved clean HTML report: {target_file}")

    root_copy = os.path.join(os.getcwd(), 'lab5_report.html')
    if root_copy != target_file:
        try:
            with open(root_copy, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"[✓] Created report copy at: {root_copy}")
        except Exception:
            pass

    return target_file
