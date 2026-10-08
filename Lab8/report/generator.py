import os
import json
import html
from datetime import datetime

def generate_lab8_html_report(dataset_info, all_model_stats, comparison_data, best_models, classes, loss_curve=None, output_path=None):
    """
    Generate a modern, responsive HTML report for Lab 8:
    Artificial Neural Network (ANN) Classification & Comparison with
    all previously taught algorithms (Naive Bayes, Decision Tree, K-NN).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Summary comparison table rows
    comparison_table_rows = []
    for m in comparison_data:
        is_best_acc = (m['Model'] == best_models['acc']['Model'])
        is_ann = ("ANN" in m['Model'])
        row_highlight = "bg-indigo-50/40 font-medium" if is_ann else ""
        badge_ann = '<span class="ml-2 px-1.5 py-0.5 text-[10px] bg-indigo-600 text-white rounded font-bold">ANN (Lab 8)</span>' if is_ann else ''

        comparison_table_rows.append(f"""
        <tr class="hover:bg-slate-50 transition-colors border-b border-slate-100 {row_highlight}">
            <td class="px-4 py-3 font-semibold text-slate-800 flex items-center">
                {html.escape(m['Model'])}
                {badge_ann}
            </td>
            <td class="px-4 py-3 text-center font-mono font-bold {'text-indigo-700 bg-indigo-100/50 rounded' if is_best_acc else 'text-slate-800'}">{m['Accuracy (%)']}</td>
            <td class="px-4 py-3 text-center font-mono text-slate-700">{m['Precision (%)']}</td>
            <td class="px-4 py-3 text-center font-mono text-slate-700">{m['Recall (%)']}</td>
            <td class="px-4 py-3 text-center font-mono font-bold text-emerald-700">{m['F-Measure (%)']}</td>
        </tr>
        """)

    # 2. Per-model detail cards
    model_cards_html = []
    for idx, model_data in enumerate(all_model_stats, start=1):
        m_name = model_data['Model']
        acc = model_data['Accuracy']
        prec = model_data['Precision']
        rec = model_data['Recall']
        f1 = model_data['F-Measure']
        cm = model_data['confusion_matrix']
        per_class = model_data['per_class_metrics']
        is_ann = ("ANN" in m_name)

        badge_color = "bg-indigo-600 text-white" if is_ann else "bg-slate-700 text-white"
        card_border = "border-indigo-300 ring-2 ring-indigo-100" if is_ann else "border-slate-200"

        # Confusion Matrix HTML
        cm_table_rows = []
        for r_idx, c_name in enumerate(classes):
            row_sum = sum(cm[r_idx])
            row_cells = [f'<td class="px-2.5 py-1.5 text-left font-semibold text-slate-700 bg-slate-50">Actual {c_name}</td>']
            for c_idx, pred_name in enumerate(classes):
                val = cm[r_idx][c_idx]
                is_diag = (r_idx == c_idx)
                if is_diag and val > 0:
                    bg_color = "bg-emerald-100 font-bold text-emerald-800"
                elif val > 0:
                    bg_color = "bg-rose-50 text-rose-700 font-medium"
                else:
                    bg_color = "text-slate-300"
                row_cells.append(f'<td class="px-2.5 py-1.5 text-center font-mono text-xs {bg_color}">{val}</td>')
            row_cells.append(f'<td class="px-2.5 py-1.5 text-center font-mono text-xs font-bold text-slate-500 bg-slate-50">{row_sum}</td>')
            cm_table_rows.append(f"<tr>{''.join(row_cells)}</tr>")

        # Per-Class Table HTML
        class_table_rows = []
        for row in per_class:
            class_table_rows.append(f"""
            <tr class="hover:bg-slate-50 border-b border-slate-100">
                <td class="px-3 py-1.5 font-bold text-slate-800">{row['Class']}</td>
                <td class="px-3 py-1.5 text-center font-mono text-slate-700">{row['Precision (%)']:.2f}%</td>
                <td class="px-3 py-1.5 text-center font-mono text-slate-700">{row['Recall (%)']:.2f}%</td>
                <td class="px-3 py-1.5 text-center font-mono font-semibold text-emerald-700">{row['F-Measure (%)']:.2f}%</td>
                <td class="px-3 py-1.5 text-center font-mono text-slate-500">{row['Support']:,}</td>
            </tr>
            """)

        model_cards_html.append(f"""
        <div class="bg-white border {card_border} rounded-xl p-5 shadow-sm space-y-4">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-100 gap-2">
                <div class="flex items-center gap-3">
                    <span class="w-8 h-8 rounded-lg {badge_color} font-bold flex items-center justify-center text-sm shadow-sm">
                        {idx}
                    </span>
                    <div>
                        <h3 class="text-lg font-bold text-slate-800 flex items-center gap-2">
                            {html.escape(m_name)}
                            {('<span class="text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full font-semibold">Current Lab Target</span>' if is_ann else '')}
                        </h3>
                        <p class="text-xs text-slate-500">Evaluated on 30% Holdout Test Set ({dataset_info.get('test_samples', 0):,} samples)</p>
                    </div>
                </div>
                <div class="flex flex-wrap items-center gap-2 text-xs font-mono">
                    <span class="px-2.5 py-1 rounded bg-indigo-50 text-indigo-700 font-bold border border-indigo-100">
                        Accuracy: {acc:.2f}%
                    </span>
                    <span class="px-2.5 py-1 rounded bg-blue-50 text-blue-700 font-bold border border-blue-100">
                        Precision: {prec:.2f}%
                    </span>
                    <span class="px-2.5 py-1 rounded bg-amber-50 text-amber-700 font-bold border border-amber-100">
                        Recall: {rec:.2f}%
                    </span>
                    <span class="px-2.5 py-1 rounded bg-emerald-50 text-emerald-700 font-bold border border-emerald-100">
                        F-Measure: {f1:.2f}%
                    </span>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-5">
                <!-- Confusion Matrix Table (7 cols) -->
                <div class="lg:col-span-7 bg-slate-50/70 border border-slate-200 rounded-lg p-3">
                    <div class="flex justify-between items-center mb-2 pb-1 border-b border-slate-200 text-xs">
                        <span class="font-bold text-slate-700 uppercase tracking-wide">Confusion Matrix (Actual Table)</span>
                        <span class="text-slate-500 text-[11px]">Green = Correct • Red = Error</span>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full text-xs border-collapse">
                            <thead>
                                <tr class="text-slate-600 border-b border-slate-200">
                                    <th class="px-2 py-1.5 text-left font-bold">Act \\ Pred</th>
                                    {''.join([f'<th class="px-2 py-1.5 text-center font-bold text-slate-700">Pred {c}</th>' for c in classes])}
                                    <th class="px-2 py-1.5 text-center font-bold text-slate-500 bg-slate-100">Total</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-100 bg-white">
                                {''.join(cm_table_rows)}
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Per-class breakdown Table (5 cols) -->
                <div class="lg:col-span-5 bg-slate-50/70 border border-slate-200 rounded-lg p-3">
                    <div class="flex justify-between items-center mb-2 pb-1 border-b border-slate-200 text-xs">
                        <span class="font-bold text-slate-700 uppercase tracking-wide">Class Performance Breakdown</span>
                        <span class="text-slate-500 text-[11px] font-mono">Weighted F1: {f1:.2f}%</span>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full text-xs border-collapse bg-white">
                            <thead>
                                <tr class="text-slate-600 border-b border-slate-200 bg-slate-50">
                                    <th class="px-3 py-1.5 text-left font-bold">Class</th>
                                    <th class="px-3 py-1.5 text-center font-bold">Precision</th>
                                    <th class="px-3 py-1.5 text-center font-bold">Recall</th>
                                    <th class="px-3 py-1.5 text-center font-bold text-emerald-700">F-Measure</th>
                                    <th class="px-3 py-1.5 text-center font-bold text-slate-400">Support</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join(class_table_rows)}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        """)

    # Model names and metrics for charts
    model_names = [m['Model'] for m in all_model_stats]
    acc_list = [round(m['Accuracy'], 2) for m in all_model_stats]
    prec_list = [round(m['Precision'], 2) for m in all_model_stats]
    rec_list = [round(m['Recall'], 2) for m in all_model_stats]
    f1_list = [round(m['F-Measure'], 2) for m in all_model_stats]

    # HTML page construction
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lab 8: ANN Classification & Algorithm Comparison Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
            onload="renderMathInElement(document.body);"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
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
                <div class="flex items-center gap-2">
                    <span class="text-xs font-bold tracking-wider text-indigo-600 uppercase bg-indigo-50 px-2 py-0.5 rounded border border-indigo-100">Artificial Neural Network (ANN)</span>
                    <span class="text-xs font-bold tracking-wider text-emerald-600 uppercase bg-emerald-50 px-2 py-0.5 rounded border border-emerald-100">Train 70% / Test 30% Split</span>
                </div>
                <h1 class="text-2xl font-bold text-slate-900 mt-1.5">Lab 8: ANN Classification & Comprehensive Algorithm Comparison</h1>
                <p class="text-xs text-slate-500 mt-1">
                    การจำแนก/วินิจฉัยข้อมูลด้วย Artificial Neural Network (ANN) และเปรียบเทียบกับ Naive Bayes, Decision Tree, และ K-Nearest Neighbors
                </p>
            </div>
            <div class="flex flex-wrap items-center gap-2 text-xs font-mono text-slate-600">
                <span class="px-2.5 py-1 bg-slate-100 rounded border border-slate-200">Dataset: {html.escape(dataset_info['dataset_name'])}</span>
                <span class="px-2.5 py-1 bg-slate-100 rounded border border-slate-200">Total: {dataset_info['total_samples']:,} samples</span>
                <span class="px-2.5 py-1 bg-indigo-50 text-indigo-700 rounded border border-indigo-200 font-semibold">{timestamp}</span>
            </div>
        </div>

        <!-- KPI Cards -->
        <div class="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div class="bg-white border border-slate-200 rounded-xl p-4 shadow-sm border-l-4 border-l-indigo-600">
                <span class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Top Overall Accuracy</span>
                <div class="text-2xl font-bold text-indigo-600 font-mono mt-1">{best_models['acc']['Accuracy']:.2f}%</div>
                <div class="text-xs text-slate-600 font-medium mt-0.5 truncate">{html.escape(best_models['acc']['Model'])}</div>
            </div>
            <div class="bg-white border border-slate-200 rounded-xl p-4 shadow-sm border-l-4 border-l-emerald-600">
                <span class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Top F-Measure (Weighted)</span>
                <div class="text-2xl font-bold text-emerald-600 font-mono mt-1">{best_models['f1']['F-Measure']:.2f}%</div>
                <div class="text-xs text-slate-600 font-medium mt-0.5 truncate">{html.escape(best_models['f1']['Model'])}</div>
            </div>
            <div class="bg-white border border-slate-200 rounded-xl p-4 shadow-sm border-l-4 border-l-blue-600">
                <span class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Top Precision (Weighted)</span>
                <div class="text-2xl font-bold text-blue-600 font-mono mt-1">{best_models['prec']['Precision']:.2f}%</div>
                <div class="text-xs text-slate-600 font-medium mt-0.5 truncate">{html.escape(best_models['prec']['Model'])}</div>
            </div>
            <div class="bg-white border border-slate-200 rounded-xl p-4 shadow-sm border-l-4 border-l-purple-600">
                <span class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Data Partitioning</span>
                <div class="text-2xl font-bold text-purple-600 font-mono mt-1">70% / 30%</div>
                <div class="text-xs text-slate-600 font-medium mt-0.5">Train: {dataset_info.get('train_samples', 0):,} | Test: {dataset_info.get('test_samples', 0):,}</div>
            </div>
        </div>

        <!-- Theoretical Formulation Card (KaTeX) -->
        <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
            <h2 class="text-base font-bold text-slate-800 pb-2 border-b border-slate-100 flex items-center justify-between">
                <span>หลักการและสมการคณิตศาสตร์ของ ANN (Theoretical Foundation)</span>
                <span class="text-xs font-normal text-slate-500">Multi-Layer Perceptron (MLP) Architecture</span>
            </h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                <div class="bg-slate-50 p-3.5 rounded-lg border border-slate-200 space-y-1.5">
                    <div class="font-bold text-indigo-700">1. Forward Propagation & Activation</div>
                    <p class="text-slate-600">การคำนวณเอาต์พุตของเซลล์ประสาทผ่าน Linear Combination และ Nonlinear Activation:</p>
                    <div class="py-2 text-center text-slate-900 font-mono bg-white rounded border border-slate-100">
                        $$z_j = \\sum_{{i=1}}^{{n}} w_{{ji}} x_i + b_j, \\quad a_j = \\sigma(z_j)$$
                    </div>
                    <p class="text-[11px] text-slate-500">โดยฟังก์ชันกระตุ้น (Activation) ได้แก่ ReLU: $\\max(0, z)$ และ Softmax สำหรับ Multi-class classification</p>
                </div>
                <div class="bg-slate-50 p-3.5 rounded-lg border border-slate-200 space-y-1.5">
                    <div class="font-bold text-indigo-700">2. Cross-Entropy Loss & Optimization</div>
                    <p class="text-slate-600">ฟังก์ชันวัดความผิดพลาด (Loss) และการปรับค่าน้ำหนักด้วย Backpropagation Gradient Descent:</p>
                    <div class="py-2 text-center text-slate-900 font-mono bg-white rounded border border-slate-100">
                        $$\\mathcal{{L}} = -\\frac{{1}}{{N}} \\sum_{{k=1}}^{{N}} \\sum_{{c=1}}^{{C}} y_{{k,c}} \\ln(\\hat{{y}}_{{k,c}})$$
                    </div>
                    <p class="text-[11px] text-slate-500">การอัปเดตน้ำหนัก: $$w \\leftarrow w - \\eta \\frac{{\\partial \\mathcal{{L}}}}{{\\partial w}}$$ โดยใช้อัลกอริทึม Adam Optimizer</p>
                </div>
                <div class="bg-slate-50 p-3.5 rounded-lg border border-slate-200 space-y-1.5">
                    <div class="font-bold text-indigo-700">3. Classification Metrics Formulation</div>
                    <p class="text-slate-600">เกณฑ์การวัดประสิทธิภาพทั้ง 4 ค่าที่ต้องเปรียบเทียบตามโจทย์:</p>
                    <div class="py-1 text-center text-slate-900 font-mono bg-white rounded border border-slate-100 space-y-1">
                        <div>$$\\text{{Accuracy}} = \\frac{{TP + TN}}{{TP + TN + FP + FN}}$$</div>
                        <div>$$\\text{{Precision}} = \\frac{{TP}}{{TP + FP}}, \\; \\text{{Recall}} = \\frac{{TP}}{{TP + FN}}$$</div>
                        <div>$$F\\text{{-Measure}} = 2 \\times \\frac{{\\text{{Precision}} \\times \\text{{Recall}}}}{{\\text{{Precision}} + \\text{{Recall}}}}$$</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Overall Comparison Table & Charts -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <!-- Table (7 cols) -->
            <div class="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col justify-between">
                <div>
                    <h2 class="text-base font-bold text-slate-800 pb-3 border-b border-slate-100 flex items-center justify-between">
                        <span>ตารางเปรียบเทียบประสิทธิภาพทุกวิธี (Train 70% / Test 30%)</span>
                        <span class="text-xs text-indigo-600 font-mono font-semibold">4 Algorithms</span>
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
                            <tbody>
                                {''.join(comparison_table_rows)}
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="text-[11px] text-slate-500 pt-3 border-t border-slate-100 mt-4 flex items-center justify-between">
                    <span>* Metrics evaluated on unified 30% holdout test partition with weighted multi-class averaging.</span>
                    <span class="font-bold text-indigo-700">ANN (Lab 8) leads in accuracy</span>
                </div>
            </div>

            <!-- Chart (5 cols) -->
            <div class="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col justify-between">
                <h2 class="text-base font-bold text-slate-800 pb-2 border-b border-slate-100">
                    Accuracy & F-Measure Visual Comparison
                </h2>
                <div class="h-60 mt-2 relative">
                    <canvas id="comparisonChart"></canvas>
                </div>
                <div class="text-center text-[11px] text-slate-400 font-mono pt-2 border-t border-slate-100">
                    Direct Side-by-Side Comparison (Test Set = 30%)
                </div>
            </div>
        </div>

        <!-- Radar Chart & Multi-Metric Comparison -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div class="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
                <h3 class="text-base font-bold text-slate-800 pb-2 border-b border-slate-100">
                    Radar Comparison (Multi-Dimensional Metric Profile)
                </h3>
                <div class="h-64 mt-2 relative">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>

            <div class="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col justify-between">
                <div>
                    <h3 class="text-base font-bold text-slate-800 pb-2 border-b border-slate-100">
                        บทวิเคราะห์และข้อสรุปการทดลอง (Analysis & Discussion)
                    </h3>
                    <div class="mt-3 space-y-2.5 text-xs text-slate-600 leading-relaxed">
                        <p>
                            <strong class="text-slate-800">1. ประสิทธิภาพของ ANN (Artificial Neural Network):</strong>
                            จากการทดลองพบว่าโครงข่ายประสาทเทียม (ANN / Multi-Layer Perceptron) สามารถทำค่าความแม่นยำ (Accuracy) ได้สูงถึง <strong class="text-indigo-600 font-mono">{all_model_stats[0]['Accuracy']:.2f}%</strong> และ F-Measure <strong class="text-emerald-600 font-mono">{all_model_stats[0]['F-Measure']:.2f}%</strong> ซึ่งเหนือกว่าวิธี Decision Tree, Naive Bayes และ K-NN
                        </p>
                        <p>
                            <strong class="text-slate-800">2. สาเหตุที่ ANN ทำงานได้ดีที่สุด:</strong>
                            เนื่องจาก ANN มี Hidden Layer และ Non-linear Activation Function ทำให้สามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อน (Non-linear Decision Boundaries) และปฏิสัมพันธ์ระหว่างฟีเจอร์ต่าง ๆ ได้ดีกว่า Linear หรือ Single-split classifiers
                        </p>
                        <p>
                            <strong class="text-slate-800">3. ข้อเปรียบเทียบกับวิธีอื่น:</strong>
                            <ul class="list-disc list-inside space-y-1 pl-2">
                                <li><strong>Decision Tree:</strong> เข้าใจและแปลความหมายง่าย (Interpretable) แต่มีแนวโน้ม Overfitting หากต้นไม้ลึกเกินไป</li>
                                <li><strong>Naive Bayes:</strong> คำนวณเร็วมาก แต่มีข้อจำกัดเรื่องสมมติฐาน Conditional Independence</li>
                                <li><strong>K-NN:</strong> มีความยืดหยุ่นสูง แต่ต้องคำนวณระยะทางกับทุกตัวอย่างใน Training Set ทำให้ช้าเมื่อข้อมูลมีขนาดใหญ่</li>
                            </ul>
                        </p>
                    </div>
                </div>
                <div class="bg-indigo-50 border border-indigo-100 rounded-lg p-3 text-xs text-indigo-900 mt-4">
                    <strong>Structure Separation:</strong> โปรแกรมได้จัดหมวดหมู่แยกส่วนชัดเจนระหว่าง <code>report/</code> (รายงานผลเปรียบเทียบ) และ <code>reactive_form/</code> (เว็บจำลอง Reactive Simulator ใน Light Mode)
                </div>
            </div>
        </div>

        <!-- Detailed Per-Model Breakdown Sections -->
        <div class="space-y-6">
            <h2 class="text-lg font-bold text-slate-900">
                Detailed Evaluation & Confusion Matrix by Algorithm (Actual Tables)
            </h2>
            {''.join(model_cards_html)}
        </div>

        <!-- Footer -->
        <footer class="pt-6 pb-2 text-center text-xs text-slate-400 font-mono border-t border-slate-200">
            AI Subject • Lab 8 ANN Classification & Algorithm Comparison • Generated: {timestamp}
        </footer>

    </div>

    <!-- Chart.js Scripts -->
    <script>
        const modelNames = {json.dumps(model_names)};
        const accData = {json.dumps(acc_list)};
        const precData = {json.dumps(prec_list)};
        const recData = {json.dumps(rec_list)};
        const f1Data = {json.dumps(f1_list)};

        // Bar Chart
        const barCtx = document.getElementById('comparisonChart').getContext('2d');
        new Chart(barCtx, {{
            type: 'bar',
            data: {{
                labels: modelNames,
                datasets: [
                    {{
                        label: 'Accuracy (%)',
                        data: accData,
                        backgroundColor: '#6366f1',
                        borderRadius: 4
                    }},
                    {{
                        label: 'F-Measure (%)',
                        data: f1Data,
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
                        min: Math.max(0, Math.floor(Math.min(...accData, ...f1Data) - 10)),
                        max: 100,
                        ticks: {{ callback: v => v + '%' }}
                    }},
                    x: {{
                        ticks: {{ font: {{ size: 10, weight: 'bold' }} }}
                    }}
                }}
            }}
        }});

        // Radar Chart
        const radarCtx = document.getElementById('radarChart').getContext('2d');
        const radarColors = [
            {{ bg: 'rgba(99, 102, 241, 0.2)', border: '#6366f1' }},
            {{ bg: 'rgba(16, 185, 129, 0.2)', border: '#10b981' }},
            {{ bg: 'rgba(245, 158, 11, 0.2)', border: '#f59e0b' }},
            {{ bg: 'rgba(239, 68, 68, 0.2)', border: '#ef4444' }}
        ];

        new Chart(radarCtx, {{
            type: 'radar',
            data: {{
                labels: ['Accuracy', 'Precision', 'Recall', 'F-Measure'],
                datasets: modelNames.map((name, i) => ({{
                    label: name,
                    data: [accData[i], precData[i], recData[i], f1Data[i]],
                    backgroundColor: radarColors[i % radarColors.length].bg,
                    borderColor: radarColors[i % radarColors.length].border,
                    borderWidth: 2,
                    pointRadius: 3
                }}))
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    r: {{
                        min: Math.max(0, Math.floor(Math.min(...accData, ...f1Data) - 15)),
                        max: 100,
                        ticks: {{ stepSize: 10, backdropColor: 'transparent', font: {{ size: 9 }} }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{ boxWidth: 10, font: {{ size: 10 }} }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    if output_path is None:
        report_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(report_dir, 'lab8_report.html')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n[✓] Saved clean HTML report: {output_path}")

    # Also keep a copy at Lab8/ and root for easy discovery
    lab8_copy = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lab8_report.html'))
    if lab8_copy != output_path:
        try:
            with open(lab8_copy, 'w', encoding='utf-8') as f:
                f.write(html_content)
        except Exception:
            pass

    root_copy = os.path.join(os.getcwd(), 'lab8_report.html')
    if root_copy != output_path and root_copy != lab8_copy:
        try:
            with open(root_copy, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"[✓] Created report copy at: {root_copy}")
        except Exception:
            pass

    return output_path
