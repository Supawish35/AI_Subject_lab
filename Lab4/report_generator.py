import os
import json
import html
from datetime import datetime

def generate_html_report_file(eval_results, preset_results, last_custom_result, feature_names, dataset_name, current_k, output_path=None):
    """
    Generate modern, responsive HTML report according to UI/UX best practices.
    All labels and text in English.
    """
    # Calculate KPIs
    acc_pct = eval_results['accuracy'] * 100
    rep = eval_results['report_dict']
    weighted_f1 = rep.get('weighted avg', {}).get('f1-score', 0) * 100
    weighted_prec = rep.get('weighted avg', {}).get('precision', 0) * 100
    weighted_rec = rep.get('weighted avg', {}).get('recall', 0) * 100
    macro_f1 = rep.get('macro avg', {}).get('f1-score', 0) * 100
    macro_prec = rep.get('macro avg', {}).get('precision', 0) * 100
    macro_rec = rep.get('macro avg', {}).get('recall', 0) * 100

    # Build Confusion Matrix rows
    classes = eval_results['classes']
    cm = eval_results['confusion_matrix']

    cm_max = max(max(row) for row in cm) if cm else 1
    cm_html_rows = []
    for r_idx, c_name in enumerate(classes):
        row_sum = sum(cm[r_idx])
        tds = [f'<td class="font-medium text-left px-3 py-2 text-surface-200">Actual {c_name}</td>']
        for c_idx, pred_c in enumerate(classes):
            val = cm[r_idx][c_idx]
            pct = (val / row_sum * 100) if row_sum > 0 else 0
            is_diag = (r_idx == c_idx)
            if is_diag:
                opacity = min(0.9, max(0.18, val / cm_max))
                style = f"background: rgba(16, 185, 129, {opacity:.2f}); color: #ffffff;"
            elif val > 0:
                opacity = min(0.7, max(0.12, (val / cm_max) * 2))
                style = f"background: rgba(239, 68, 68, {opacity:.2f}); color: #ffffff;"
            else:
                style = "background: rgba(255, 255, 255, 0.02); color: var(--text-muted);"
            tds.append(f'<td style="{style}" class="text-center font-mono font-bold py-2.5 px-3 rounded-sm transition-all">{val}<div class="text-[10px] opacity-75">{pct:.1f}%</div></td>')
        tds.append(f'<td class="font-mono text-center px-3 py-2 text-surface-400 font-semibold">{row_sum:,}</td>')
        cm_html_rows.append(f"<tr>{''.join(tds)}</tr>")

    # Classification Report Table Rows
    report_html_rows = []
    for c in classes:
        c_data = rep.get(c, {})
        p = c_data.get('precision', 0) * 100
        r = c_data.get('recall', 0) * 100
        f = c_data.get('f1-score', 0) * 100
        s = int(c_data.get('support', 0))
        badge_colors = {
            'A': 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
            'B': 'bg-blue-500/15 text-blue-400 border-blue-500/30',
            'C': 'bg-amber-500/15 text-amber-400 border-amber-500/30',
            'D': 'bg-orange-500/15 text-orange-400 border-orange-500/30',
            'F': 'bg-rose-500/15 text-rose-400 border-rose-500/30'
        }
        badge_cls = badge_colors.get(c, 'bg-slate-500/15 text-slate-300 border-slate-500/30')
        report_html_rows.append(f"""
        <tr class="hover:bg-surface-800/40 border-b border-surface-700/50 transition-colors">
            <td class="py-3 px-4">
                <span class="inline-flex items-center justify-center w-8 h-8 rounded-lg font-bold border {badge_cls}">
                    {c}
                </span>
            </td>
            <td class="py-3 px-4 font-mono">
                <div class="flex items-center gap-2">
                    <span class="w-14 text-right font-semibold">{p:.1f}%</span>
                    <div class="flex-1 bg-surface-700/60 rounded-full h-2 overflow-hidden max-w-[100px]">
                        <div class="bg-blue-500 h-full rounded-full" style="width: {p}%"></div>
                    </div>
                </div>
            </td>
            <td class="py-3 px-4 font-mono">
                <div class="flex items-center gap-2">
                    <span class="w-14 text-right font-semibold">{r:.1f}%</span>
                    <div class="flex-1 bg-surface-700/60 rounded-full h-2 overflow-hidden max-w-[100px]">
                        <div class="bg-emerald-500 h-full rounded-full" style="width: {r}%"></div>
                    </div>
                </div>
            </td>
            <td class="py-3 px-4 font-mono">
                <div class="flex items-center gap-2">
                    <span class="w-14 text-right font-semibold">{f:.1f}%</span>
                    <div class="flex-1 bg-surface-700/60 rounded-full h-2 overflow-hidden max-w-[100px]">
                        <div class="bg-purple-500 h-full rounded-full" style="width: {f}%"></div>
                    </div>
                </div>
            </td>
            <td class="py-3 px-4 font-mono text-right text-surface-400 font-semibold">{s:,}</td>
        </tr>
        """)

    # Presets HTML Cards
    preset_cards_html = []
    for p_idx, pres in enumerate(preset_results, 1):
        pred_grade = pres['predicted_class']
        grade_badges = {
            'A': 'bg-emerald-500 text-white shadow-emerald-500/25',
            'B': 'bg-blue-500 text-white shadow-blue-500/25',
            'C': 'bg-amber-500 text-slate-900 shadow-amber-500/25',
            'D': 'bg-orange-500 text-white shadow-orange-500/25',
            'F': 'bg-rose-500 text-white shadow-rose-500/25'
        }
        g_badge = grade_badges.get(pred_grade, 'bg-primary-500 text-white')

        nn_rows = []
        for nn in pres['nearest_neighbors']:
            c_tag = nn['class']
            nn_rows.append(f"""
            <tr class="border-b border-surface-800 text-xs">
                <td class="py-1.5 px-2 text-surface-400">#{nn['rank']}</td>
                <td class="py-1.5 px-2 font-mono text-surface-300">ID #{nn['index']}</td>
                <td class="py-1.5 px-2 font-mono text-right text-accent-400 font-semibold">{nn['distance']:.4f}</td>
                <td class="py-1.5 px-2 text-center font-bold">
                    <span class="px-2 py-0.5 rounded text-[10px] bg-surface-800 border border-surface-700 text-surface-200">
                        {c_tag}
                    </span>
                </td>
            </tr>
            """)

        vote_bars = []
        total_v = pres['total_votes']
        for c_vote, v_cnt in pres['vote_counts'].items():
            v_pct = (v_cnt / total_v) * 100
            is_win = (c_vote == pred_grade)
            bar_color = "bg-primary-500" if is_win else "bg-surface-600"
            vote_bars.append(f"""
            <div class="flex items-center gap-2 text-xs mb-1.5">
                <span class="w-6 font-bold text-surface-300">Grade {c_vote}</span>
                <div class="flex-1 bg-surface-800 rounded-full h-2.5 overflow-hidden">
                    <div class="{bar_color} h-full rounded-full transition-all" style="width: {v_pct}%"></div>
                </div>
                <span class="w-12 text-right font-mono text-surface-300">{v_cnt}/{total_v} ({v_pct:.0f}%)</span>
            </div>
            """)

        raw_feat_chips = []
        for k_raw, v_raw in pres['raw_inputs'].items():
            raw_feat_chips.append(f"""
            <div class="bg-surface-800/80 border border-surface-700/60 rounded px-2.5 py-1 text-xs flex justify-between gap-2">
                <span class="text-surface-400">{k_raw}:</span>
                <span class="font-mono font-medium text-surface-200">{v_raw}</span>
            </div>
            """)

        preset_cards_html.append(f"""
        <div class="bg-surface-900/90 border border-surface-700/70 rounded-xl p-5 shadow-lg backdrop-blur flex flex-col gap-4">
            <div class="flex items-start justify-between gap-4 pb-3 border-b border-surface-800">
                <div>
                    <span class="text-xs font-semibold uppercase tracking-wider text-primary-400">Case Study {p_idx}</span>
                    <h3 class="text-lg font-bold text-white mt-0.5">{html.escape(pres['title'])}</h3>
                </div>
                <div class="text-right">
                    <span class="text-[11px] text-surface-400 uppercase font-semibold block">Predicted Grade</span>
                    <span class="inline-flex items-center justify-center w-11 h-11 rounded-xl text-xl font-black shadow-lg {g_badge} mt-1">
                        {pred_grade}
                    </span>
                </div>
            </div>

            <div>
                <h4 class="text-xs font-semibold text-surface-400 uppercase mb-2">Input Student Attributes</h4>
                <div class="grid grid-cols-2 gap-2">
                    {''.join(raw_feat_chips)}
                </div>
            </div>

            <div>
                <h4 class="text-xs font-semibold text-surface-400 uppercase mb-2">Majority Vote Breakdown (K = {pres['k']})</h4>
                {''.join(vote_bars)}
            </div>

            <div>
                <h4 class="text-xs font-semibold text-surface-400 uppercase mb-2">Nearest Neighbors Details</h4>
                <div class="overflow-x-auto rounded border border-surface-800">
                    <table class="w-full text-left">
                        <thead>
                            <tr class="bg-surface-800/80 text-[11px] text-surface-400 font-semibold">
                                <th class="py-1 px-2">Rank</th>
                                <th class="py-1 px-2">Sample</th>
                                <th class="py-1 px-2 text-right">Distance</th>
                                <th class="py-1 px-2 text-center">Class</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(nn_rows)}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """)

    # Custom prediction card (if available)
    custom_card_html = ""
    if last_custom_result:
        c_res = last_custom_result
        custom_card_html = f"""
        <div class="mt-8 bg-surface-900 border-2 border-primary-500/50 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
            <div class="absolute -right-10 -top-10 w-40 h-40 bg-primary-500/10 rounded-full blur-3xl pointer-events-none"></div>
            <div class="flex items-center justify-between mb-4">
                <div>
                    <span class="inline-block px-2.5 py-0.5 rounded-full text-xs font-bold bg-primary-500/20 text-primary-400 border border-primary-500/30 mb-1">Interactive User Session</span>
                    <h3 class="text-xl font-bold text-white">Custom Tested Student Prediction Result</h3>
                </div>
                <div class="flex items-center gap-3">
                    <span class="text-sm text-surface-400">Predicted Class:</span>
                    <span class="w-12 h-12 rounded-xl bg-emerald-500 text-white font-black text-2xl flex items-center justify-center shadow-lg shadow-emerald-500/30">
                        {c_res['predicted_class']}
                    </span>
                </div>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-surface-800">
                <div>
                    <h4 class="text-xs font-semibold text-surface-400 uppercase mb-3">Custom Input Parameters</h4>
                    <div class="space-y-1.5">
                        {''.join([f'<div class="flex justify-between text-xs py-1 px-2 rounded bg-surface-800/60"><span class="text-surface-400">{k}:</span><span class="font-mono text-surface-200 font-bold">{v}</span></div>' for k, v in c_res['raw_inputs'].items()])}
                    </div>
                </div>
                <div>
                    <h4 class="text-xs font-semibold text-surface-400 uppercase mb-3">K Nearest Neighbors (K = {c_res['k']})</h4>
                    <div class="space-y-1.5">
                        {''.join([f'<div class="flex justify-between items-center text-xs py-1.5 px-3 rounded bg-surface-800/70 border border-surface-700/50 font-mono"><span>Rank #{nn["rank"]} (ID #{nn["index"]})</span><span class="text-accent-400">dist: {nn["distance"]:.4f}</span><span class="px-2 py-0.5 rounded bg-surface-700 text-white font-bold">{nn["class"]}</span></div>' for nn in c_res['nearest_neighbors']])}
                    </div>
                </div>
            </div>
        </div>
        """

    # Assemble HTML content
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lab 4: K-Nearest Neighbors Classification Executive Report</title>
    <!-- Tailwind CSS v3 CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {{
            darkMode: 'class',
            theme: {{
                extend: {{
                    colors: {{
                        primary: {{
                            50: '#eef2ff',
                            100: '#e0e7ff',
                            400: '#818cf8',
                            500: '#6366f1',
                            600: '#4f46e5',
                            700: '#4338ca'
                        }},
                        accent: {{
                            400: '#38bdf8',
                            500: '#0ea5e9'
                        }},
                        surface: {{
                            950: '#07090e',
                            900: '#0f172a',
                            850: '#131e36',
                            800: '#1e293b',
                            700: '#334155',
                            600: '#475569',
                            400: '#94a3b8',
                            300: '#cbd5e1',
                            200: '#e2e8f0',
                            100: '#f1f5f9'
                        }}
                    }},
                    fontFamily: {{
                        sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
                        mono: ['"JetBrains Mono"', 'monospace']
                    }}
                }}
            }}
        }}
    </script>
    <style>
        :root {{
            --bg-base: #07090e;
            --surface-card: #0f172a;
            --border-dim: #1e293b;
            --text-muted: #64748b;
        }}
        body {{
            background-color: var(--bg-base);
            color: #f8fafc;
            font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
            -webkit-font-smoothing: antialiased;
        }}
        .glass-panel {{
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .glow-accent {{
            box-shadow: 0 0 40px -10px rgba(99, 102, 241, 0.25);
        }}
    </style>
</head>
<body class="min-h-screen text-surface-100 flex flex-col">

    <!-- Top Navigation Bar -->
    <header class="sticky top-0 z-50 glass-panel border-b border-surface-800">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center gap-3">
                <div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-primary-600 to-accent-500 flex items-center justify-center font-black text-white text-lg shadow-lg shadow-primary-500/20">
                    K
                </div>
                <div>
                    <h1 class="text-base font-bold text-white tracking-tight leading-none">Lab 4: K-NN Classifier</h1>
                    <span class="text-[11px] text-surface-400">Machine Learning & Artificial Intelligence</span>
                </div>
            </div>
            <div class="flex items-center gap-3 text-xs font-mono text-surface-400">
                <span class="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-surface-800/80 border border-surface-700">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    Dataset: {html.escape(dataset_name)}
                </span>
                <span class="px-2.5 py-1 rounded-full bg-primary-500/10 text-primary-400 border border-primary-500/20 font-bold">
                    K = {current_k} Neighbors
                </span>
            </div>
        </div>
    </header>

    <!-- Main Content Container -->
    <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

        <!-- Executive Hero Banner -->
        <section class="glass-panel glow-accent rounded-3xl p-6 sm:p-8 relative overflow-hidden border border-surface-700/60">
            <div class="absolute -right-20 -top-20 w-80 h-80 bg-primary-600/15 rounded-full blur-3xl pointer-events-none"></div>
            <div class="absolute -left-20 -bottom-20 w-80 h-80 bg-accent-500/10 rounded-full blur-3xl pointer-events-none"></div>

            <div class="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div class="space-y-2 max-w-2xl">
                    <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-primary-500/15 text-primary-300 border border-primary-500/30">
                        <span>Pure Vectorized Algorithm (NumPy)</span>
                        <span>•</span>
                        <span>Euclidean Distance Voting</span>
                    </div>
                    <h2 class="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                        Student Academic Performance Classification
                    </h2>
                    <p class="text-sm text-surface-300 leading-relaxed">
                        Official evaluation report using non-parametric K-Nearest Neighbors on normalized student features with multi-class letter grade prediction (A, B, C, D, F).
                    </p>
                </div>

                <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                    <div class="bg-surface-850/80 border border-surface-700/60 rounded-2xl p-3.5 min-w-[110px]">
                        <span class="text-[11px] uppercase tracking-wider text-surface-400 font-semibold block">Accuracy</span>
                        <span class="text-2xl font-black text-emerald-400 font-mono mt-0.5 block">{acc_pct:.2f}%</span>
                        <span class="text-[10px] text-surface-400">Test Set</span>
                    </div>
                    <div class="bg-surface-850/80 border border-surface-700/60 rounded-2xl p-3.5 min-w-[110px]">
                        <span class="text-[11px] uppercase tracking-wider text-surface-400 font-semibold block">Weighted F1</span>
                        <span class="text-2xl font-black text-purple-400 font-mono mt-0.5 block">{weighted_f1:.2f}%</span>
                        <span class="text-[10px] text-surface-400">Macro: {macro_f1:.1f}%</span>
                    </div>
                    <div class="bg-surface-850/80 border border-surface-700/60 rounded-2xl p-3.5 min-w-[110px]">
                        <span class="text-[11px] uppercase tracking-wider text-surface-400 font-semibold block">Precision</span>
                        <span class="text-2xl font-black text-blue-400 font-mono mt-0.5 block">{weighted_prec:.2f}%</span>
                        <span class="text-[10px] text-surface-400">Macro: {macro_prec:.1f}%</span>
                    </div>
                    <div class="bg-surface-850/80 border border-surface-700/60 rounded-2xl p-3.5 min-w-[110px]">
                        <span class="text-[11px] uppercase tracking-wider text-surface-400 font-semibold block">Neighbors</span>
                        <span class="text-2xl font-black text-primary-400 font-mono mt-0.5 block">K = {current_k}</span>
                        <span class="text-[10px] text-surface-400">{eval_results['train_samples']:,} Trains</span>
                    </div>
                </div>
            </div>
        </section>

        <!-- Evaluation Performance & Confusion Matrix -->
        <section class="grid grid-cols-1 lg:grid-cols-12 gap-8">

            <!-- Confusion Matrix Panel (7 cols) -->
            <div class="lg:col-span-7 glass-panel rounded-2xl p-6 border border-surface-700/70 flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between pb-4 border-b border-surface-800">
                        <div>
                            <h3 class="text-lg font-bold text-white">Confusion Matrix (Actual vs Predicted)</h3>
                            <p class="text-xs text-surface-400 mt-0.5">Cell intensities show prediction frequency and distribution percentage</p>
                        </div>
                        <span class="text-xs font-mono px-2.5 py-1 rounded bg-surface-800 text-surface-300 font-semibold">
                            Total: {eval_results['test_samples']:,} samples
                        </span>
                    </div>

                    <div class="overflow-x-auto mt-5">
                        <table class="w-full border-collapse text-xs">
                            <thead>
                                <tr class="text-surface-400 border-b border-surface-700/60">
                                    <th class="py-2 px-3 text-left font-semibold">Actual \\ Pred</th>
                                    {''.join([f'<th class="py-2 px-3 text-center font-bold text-primary-300">Pred {c}</th>' for c in classes])}
                                    <th class="py-2 px-3 text-center font-semibold text-surface-400">Total</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-surface-800/40">
                                {''.join(cm_html_rows)}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="mt-4 pt-4 border-t border-surface-800 flex items-center justify-between text-xs text-surface-400">
                    <div class="flex items-center gap-3">
                        <span class="inline-flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-emerald-500/80 inline-block"></span> Correct Predictions</span>
                        <span class="inline-flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-rose-500/60 inline-block"></span> Misclassifications</span>
                    </div>
                    <span class="font-mono">Overall Accuracy: <strong>{acc_pct:.2f}%</strong></span>
                </div>
            </div>

            <!-- Per-Class Metrics Bar Chart Panel (5 cols) -->
            <div class="lg:col-span-5 glass-panel rounded-2xl p-6 border border-surface-700/70 flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between pb-4 border-b border-surface-800">
                        <div>
                            <h3 class="text-lg font-bold text-white">Performance Metrics by Class</h3>
                            <p class="text-xs text-surface-400 mt-0.5">Precision, Recall, and F1 across each letter grade</p>
                        </div>
                    </div>
                    <div class="h-64 mt-4 relative">
                        <canvas id="classMetricsChart"></canvas>
                    </div>
                </div>
                <div class="text-[11px] text-surface-400 pt-3 border-t border-surface-800 text-center font-mono">
                    Evaluation completed using Stratified Test Split (20% holdout)
                </div>
            </div>
        </section>

        <!-- Detailed Classification Report Table -->
        <section class="glass-panel rounded-2xl p-6 border border-surface-700/70">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-surface-800">
                <div>
                    <h3 class="text-lg font-bold text-white">Classification Performance Report</h3>
                    <p class="text-xs text-surface-400 mt-0.5">Detailed precision, recall, f1-score, and sample support for each grade</p>
                </div>
                <div class="flex items-center gap-4 text-xs font-mono text-surface-300">
                    <div>Weighted Avg: <span class="font-bold text-emerald-400">{weighted_f1:.2f}% F1</span></div>
                    <div>Macro Avg: <span class="font-bold text-purple-400">{macro_f1:.2f}% F1</span></div>
                </div>
            </div>

            <div class="overflow-x-auto mt-4">
                <table class="w-full text-left text-sm">
                    <thead>
                        <tr class="text-surface-400 border-b border-surface-700/60 text-xs font-semibold uppercase">
                            <th class="py-3 px-4">Grade Class</th>
                            <th class="py-3 px-4">Precision (%)</th>
                            <th class="py-3 px-4">Recall (%)</th>
                            <th class="py-3 px-4">F1-Score (%)</th>
                            <th class="py-3 px-4 text-right">Support (Count)</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-surface-800/60">
                        {''.join(report_html_rows)}
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Case Studies & Preset Interactive Demonstrations -->
        <section class="space-y-4">
            <div class="flex items-center justify-between">
                <div>
                    <h3 class="text-xl font-extrabold text-white">Preset Case Studies (K-NN Voting Demonstration)</h3>
                    <p class="text-xs text-surface-400 mt-0.5">Tested student profiles representing distinct performance categories</p>
                </div>
                <span class="text-xs font-mono px-3 py-1 rounded-full bg-surface-800 border border-surface-700 text-surface-300">
                    3 Benchmarked Profiles
                </span>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                {''.join(preset_cards_html)}
            </div>

            <!-- Custom session card if run interactively -->
            {custom_card_html}
        </section>

        <!-- Mathematical Foundations Reference -->
        <section class="glass-panel rounded-2xl p-6 border border-surface-700/70">
            <h3 class="text-lg font-bold text-white mb-3">K-NN Methodological Framework</h3>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-surface-300 leading-relaxed">
                <div class="bg-surface-900/60 p-4 rounded-xl border border-surface-800">
                    <h4 class="font-bold text-primary-400 text-sm mb-1.5">1. Euclidean Distance Metric</h4>
                    <p class="font-mono text-[11px] bg-surface-950 p-2 rounded border border-surface-800 text-accent-300 mb-2">
                        d(p, q) = √( ∑ (pᵢ - qᵢ)² )
                    </p>
                    <p>Features are min-max scaled into [0, 1] range to avoid dominance by variables with large variances.</p>
                </div>
                <div class="bg-surface-900/60 p-4 rounded-xl border border-surface-800">
                    <h4 class="font-bold text-primary-400 text-sm mb-1.5">2. Distance Sorting & Ranking</h4>
                    <p class="font-mono text-[11px] bg-surface-950 p-2 rounded border border-surface-800 text-accent-300 mb-2">
                        indices = np.argsort(distances)[:K]
                    </p>
                    <p>All 8,000 training observations are indexed and sorted to find the exact top-K nearest neighbors.</p>
                </div>
                <div class="bg-surface-900/60 p-4 rounded-xl border border-surface-800">
                    <h4 class="font-bold text-primary-400 text-sm mb-1.5">3. Plurality / Majority Voting</h4>
                    <p class="font-mono text-[11px] bg-surface-950 p-2 rounded border border-surface-800 text-accent-300 mb-2">
                        y_pred = argmax( Counter(classes) )
                    </p>
                    <p>The final predicted grade is awarded to the most frequent letter grade among the K nearest students.</p>
                </div>
            </div>
        </section>

    </main>

    <!-- Footer -->
    <footer class="glass-panel border-t border-surface-800 mt-12 py-6 text-center text-xs text-surface-400">
        <div class="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3 font-mono">
            <span>AI Subject • Lab 4: K-NN Classification</span>
            <span>Generated: {timestamp_str}</span>
            <span>Dataset Rows: {eval_results['train_samples'] + eval_results['test_samples']:,}</span>
        </div>
    </footer>

    <!-- Chart.js Render Script -->
    <script>
        const ctx = document.getElementById('classMetricsChart').getContext('2d');
        const classes = {json.dumps(classes)};
        const precisionData = classes.map(c => ({json.dumps([rep.get(c, {}).get('precision', 0) * 100 for c in classes])})[classes.indexOf(c)]);
        const recallData = classes.map(c => ({json.dumps([rep.get(c, {}).get('recall', 0) * 100 for c in classes])})[classes.indexOf(c)]);
        const f1Data = classes.map(c => ({json.dumps([rep.get(c, {}).get('f1-score', 0) * 100 for c in classes])})[classes.indexOf(c)]);

        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: classes.map(c => 'Grade ' + c),
                datasets: [
                    {{
                        label: 'Precision (%)',
                        data: precisionData,
                        backgroundColor: 'rgba(59, 130, 246, 0.75)',
                        borderColor: '#3b82f6',
                        borderWidth: 1.5,
                        borderRadius: 6
                    }},
                    {{
                        label: 'Recall (%)',
                        data: recallData,
                        backgroundColor: 'rgba(16, 185, 129, 0.75)',
                        borderColor: '#10b981',
                        borderWidth: 1.5,
                        borderRadius: 6
                    }},
                    {{
                        label: 'F1-Score (%)',
                        data: f1Data,
                        backgroundColor: 'rgba(168, 85, 247, 0.75)',
                        borderColor: '#a855f7',
                        borderWidth: 1.5,
                        borderRadius: 6
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'top',
                        labels: {{
                            color: '#94a3b8',
                            font: {{ family: '"Plus Jakarta Sans"', size: 11 }},
                            boxWidth: 12,
                            padding: 12
                        }}
                    }},
                    tooltip: {{
                        backgroundColor: '#0f172a',
                        titleColor: '#ffffff',
                        bodyColor: '#cbd5e1',
                        borderColor: '#334155',
                        borderWidth: 1,
                        padding: 10
                    }}
                }},
                scales: {{
                    y: {{
                        min: 0,
                        max: 100,
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{
                            color: '#64748b',
                            font: {{ family: '"JetBrains Mono"', size: 10 }},
                            callback: v => v + '%'
                        }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{
                            color: '#cbd5e1',
                            font: {{ family: '"Plus Jakarta Sans"', weight: 'bold', size: 11 }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    dest_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.getcwd(),
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    ]

    target_file = output_path
    if not target_file:
        target_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lab4_report.html')

    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n[✓] Saved HTML report successfully: {target_file}")

    root_copy = os.path.join(os.getcwd(), 'lab4_report.html')
    if root_copy != target_file:
        try:
            with open(root_copy, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"[✓] Created report copy at: {root_copy}")
        except Exception:
            pass

    return target_file
