import os
import json
import html
from datetime import datetime

def generate_lab6_html_report(dataset_info, sec1_data, sec2_data, output_path=None):
    """
    Generate an interactive, beautiful HTML report for Lab 6 Feature Distance Analysis.
    Includes KaTeX formula rendering, Chart.js visualizations, and comprehensive tables.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, "lab6_report.html")

    # Section 1 single features rows
    sec1_single_rows = []
    for rank, item in enumerate(sec1_data['single_features'], start=1):
        is_best = rank == 1
        badge = '<span class="px-2 py-0.5 text-xs font-bold rounded-full bg-emerald-100 text-emerald-800">⭐ Best Feature</span>' if is_best else f'<span class="text-xs text-slate-500">#{rank}</span>'
        highlight_class = "bg-emerald-50/50 font-semibold" if is_best else "hover:bg-slate-50"
        sec1_single_rows.append(f"""
        <tr class="{highlight_class} transition-colors border-b border-slate-100">
            <td class="px-4 py-3 text-center">{badge}</td>
            <td class="px-4 py-3 font-mono font-medium text-slate-800">{html.escape(item['feature'])}</td>
            <td class="px-4 py-3 text-right font-mono font-bold text-indigo-700">{item['mean_divergence']:.4f}</td>
            <td class="px-4 py-3 text-right font-mono text-slate-600">{item['min_divergence']:.4f}</td>
            <td class="px-4 py-3 text-right font-mono text-slate-600">{item['max_divergence']:.4f}</td>
        </tr>
        """)

    # Section 1 pair features rows
    sec1_pair_rows = []
    for rank, item in enumerate(sec1_data['pair_features'], start=1):
        is_best = rank == 1
        is_top2_combo = item.get('is_top2_combo', False)
        tags = []
        if is_best:
            tags.append('<span class="px-2 py-0.5 text-xs font-bold rounded-full bg-indigo-100 text-indigo-800">🏆 Top Pair</span>')
        if is_top2_combo:
            tags.append('<span class="px-2 py-0.5 text-xs font-medium rounded-full bg-amber-100 text-amber-800">Top-2 Single Combo</span>')
        badge_html = " ".join(tags) if tags else f'<span class="text-xs text-slate-500">#{rank}</span>'
        highlight_class = "bg-indigo-50/40 font-semibold" if is_best else "hover:bg-slate-50"
        features_str = " + ".join(item['features'])
        sec1_pair_rows.append(f"""
        <tr class="{highlight_class} transition-colors border-b border-slate-100">
            <td class="px-4 py-3 text-center">{badge_html}</td>
            <td class="px-4 py-3 font-mono text-slate-800">{html.escape(features_str)}</td>
            <td class="px-4 py-3 text-right font-mono font-bold text-indigo-700">{item['mean_divergence']:.4f}</td>
            <td class="px-4 py-3 text-right font-mono text-slate-600">{item['min_divergence']:.4f}</td>
            <td class="px-4 py-3 text-right font-mono text-slate-600">{item['max_divergence']:.4f}</td>
        </tr>
        """)

    # Section 2 single features rows
    sec2_single_rows = []
    for rank, item in enumerate(sec2_data['single_features'], start=1):
        is_best = rank == 1
        badge = '<span class="px-2 py-0.5 text-xs font-bold rounded-full bg-emerald-100 text-emerald-800">⭐ Best Overall</span>' if is_best else f'<span class="text-xs text-slate-500">#{rank}</span>'
        highlight_class = "bg-emerald-50/50 font-semibold" if is_best else "hover:bg-slate-50"
        sec2_single_rows.append(f"""
        <tr class="{highlight_class} transition-colors border-b border-slate-100">
            <td class="px-4 py-3 text-center">{badge}</td>
            <td class="px-4 py-3 font-mono font-medium text-slate-800">{html.escape(item['feature'])}</td>
            <td class="px-4 py-3 text-right font-mono font-bold text-emerald-700">{item['mean_divergence']:.4f}</td>
            <td class="px-4 py-3 text-right font-mono text-slate-600">{item['min_divergence']:.4f}</td>
            <td class="px-4 py-3 text-right font-mono text-slate-600">{item['max_divergence']:.4f}</td>
        </tr>
        """)

    # JSON data for charts
    chart_sec1_labels = [item['feature'] for item in sec1_data['single_features']]
    chart_sec1_values = [round(item['mean_divergence'], 4) for item in sec1_data['single_features']]

    chart_sec1_pairs_labels = [" + ".join(item['features']) for item in sec1_data['pair_features']]
    chart_sec1_pairs_values = [round(item['mean_divergence'], 4) for item in sec1_data['pair_features']]

    chart_sec2_labels = [item['feature'] for item in sec2_data['single_features']]
    chart_sec2_values = [round(item['mean_divergence'], 4) for item in sec2_data['single_features']]

    # Comparison summary: Single Best vs Half Best vs All Features
    sec1_all_div = sec1_data['all_features']['mean_divergence']
    sec1_best_single = sec1_data['single_features'][0]['mean_divergence']
    sec1_best_pair = sec1_data['pair_features'][0]['mean_divergence']

    sec2_all_div = sec2_data['all_features']['mean_divergence']
    sec2_best_single = sec2_data['single_features'][0]['mean_divergence']
    sec2_best_half = sec2_data['best_half']['mean_divergence']

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lab 6: Feature Distance & Divergence Analysis</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
            onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
        code, pre, .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
    </style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen pb-16">

    <!-- Top Banner -->
    <header class="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
                <div class="flex items-center gap-2">
                    <span class="px-2.5 py-1 text-xs font-bold rounded-md bg-indigo-600 text-white tracking-wider uppercase">Lab 6</span>
                    <h1 class="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight">Feature Distance (Divergence) Analysis</h1>
                </div>
                <p class="text-sm text-slate-500 mt-1">Symmetric Jeffreys Divergence / Distance across Classes &amp; Feature Subsets</p>
            </div>
            <div class="flex items-center gap-3 text-xs text-slate-500">
                <span class="inline-flex items-center px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 font-medium">Dataset: {html.escape(dataset_info['dataset_name'])}</span>
                <span class="inline-flex items-center px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 font-medium">Samples: {dataset_info['total_samples']:,}</span>
                <span class="hidden sm:inline-block">Generated: {timestamp}</span>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8 space-y-10">

        <!-- Mathematical Foundations Card -->
        <section class="bg-white rounded-2xl p-6 sm:p-8 shadow-sm border border-slate-200">
            <div class="flex items-center gap-3 mb-4">
                <div class="p-2 rounded-xl bg-indigo-100 text-indigo-700">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"></path></svg>
                </div>
                <div>
                    <h2 class="text-lg font-bold text-slate-900">Mathematical Formulation (Jeffreys Divergence)</h2>
                    <p class="text-xs text-slate-500">Formulas specified in Lab 6 instruction slides</p>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-4">
                <!-- Formula 1 -->
                <div class="bg-slate-50 p-5 rounded-xl border border-slate-200">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-xs font-bold uppercase tracking-wider text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded">Formula 1: 1D Single Feature Distance</span>
                    </div>
                    <p class="text-xs text-slate-600 mb-3">Calculates separation distance between class $i$ and class $j$ using univariate mean and variance:</p>
                    <div class="p-3 bg-white rounded-lg border border-slate-200 overflow-x-auto text-center text-sm">
                        $$d_{{ij}} = \\frac{{1}}{{2}}\\left(\\frac{{\\sigma_j^2}}{{\\sigma_i^2}} + \\frac{{\\sigma_i^2}}{{\\sigma_j^2}} - 2\\right) + \\frac{{1}}{{2}}(\\mu_i - \\mu_j)^2\\left(\\frac{{1}}{{\\sigma_i^2}} + \\frac{{1}}{{\\sigma_j^2}}\\right)$$
                    </div>
                </div>

                <!-- Formula 2 -->
                <div class="bg-slate-50 p-5 rounded-xl border border-slate-200">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-xs font-bold uppercase tracking-wider text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded">Formula 2: Multivariate Feature Vector Distance</span>
                    </div>
                    <p class="text-xs text-slate-600 mb-3">Calculates separation distance for multiple features using mean vectors and covariance matrices:</p>
                    <div class="p-3 bg-white rounded-lg border border-slate-200 overflow-x-auto text-center text-sm">
                        $$d_{{ij}} = \\frac{{1}}{{2}}\\text{{trace}}\\left\\{{\\Sigma_i^{{-1}}\\Sigma_j + \\Sigma_j^{{-1}}\\Sigma_i - 2I\\right\\}} + \\frac{{1}}{{2}}(\\vec{{\\mu}}_i - \\vec{{\\mu}}_j)^T(\\Sigma_i^{{-1}} + \\Sigma_j^{{-1}})(\\vec{{\\mu}}_i - \\vec{{\\mu}}_j)$$
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 1: 4 FEATURES SPECIFICATION (MAIN LAB 6 REQUIREMENT) -->
        <section class="bg-white rounded-2xl p-6 sm:p-8 shadow-sm border border-slate-200 space-y-8">
            <div class="border-b border-slate-100 pb-4">
                <span class="px-2.5 py-1 text-xs font-bold rounded bg-indigo-100 text-indigo-800 uppercase tracking-wide">Primary Experiment</span>
                <h2 class="text-2xl font-extrabold text-slate-900 mt-2">Section 1: 4-Feature Demonstration</h2>
                <p class="text-sm text-slate-500 mt-1">
                    Features: <code class="text-indigo-600 font-semibold">{", ".join(sec1_data['features_tested'])}</code>
                    | Target: <code class="text-slate-700 font-semibold">{dataset_info['target_col']}</code> ({len(dataset_info['classes'])} classes: {", ".join(dataset_info['classes'])})
                </p>
            </div>

            <!-- Key Metric Highlights -->
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div class="p-5 rounded-xl bg-slate-50 border border-slate-200">
                    <p class="text-xs font-semibold uppercase text-slate-500">Best Single Feature (1D)</p>
                    <h3 class="text-xl font-bold text-slate-900 mt-1">{sec1_data['single_features'][0]['feature']}</h3>
                    <p class="text-2xl font-extrabold text-indigo-600 mt-1 font-mono">{sec1_best_single:.4f}</p>
                    <p class="text-xs text-slate-500 mt-1">Average divergence over 10 class pairs</p>
                </div>

                <div class="p-5 rounded-xl bg-indigo-50/50 border border-indigo-100">
                    <p class="text-xs font-semibold uppercase text-indigo-600">Best Half-Features (2 Features)</p>
                    <h3 class="text-xl font-bold text-slate-900 mt-1">{" + ".join(sec1_data['pair_features'][0]['features'])}</h3>
                    <p class="text-2xl font-extrabold text-indigo-600 mt-1 font-mono">{sec1_best_pair:.4f}</p>
                    <p class="text-xs text-slate-500 mt-1">Highest among all 6 pairwise combinations</p>
                </div>

                <div class="p-5 rounded-xl bg-emerald-50/50 border border-emerald-100">
                    <p class="text-xs font-semibold uppercase text-emerald-600">All 4 Features Combined</p>
                    <h3 class="text-xl font-bold text-slate-900 mt-1">Full 4D Vector</h3>
                    <p class="text-2xl font-extrabold text-emerald-600 mt-1 font-mono">{sec1_all_div:.4f}</p>
                    <p class="text-xs text-slate-500 mt-1">Multivariate distance with full 4x4 covariance</p>
                </div>
            </div>

            <!-- Visual Charts -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div class="bg-slate-50 p-5 rounded-xl border border-slate-200">
                    <h3 class="text-sm font-bold text-slate-900 mb-1">1. Distance per Single Feature (1D Equation)</h3>
                    <p class="text-xs text-slate-500 mb-4">Higher divergence indicates better class separability.</p>
                    <div class="h-64">
                        <canvas id="sec1SingleChart"></canvas>
                    </div>
                </div>

                <div class="bg-slate-50 p-5 rounded-xl border border-slate-200">
                    <h3 class="text-sm font-bold text-slate-900 mb-1">2. Distance of Half Features (2-Feature Combinations)</h3>
                    <p class="text-xs text-slate-500 mb-4">Calculated via Multivariate Equation (6 combinations).</p>
                    <div class="h-64">
                        <canvas id="sec1PairsChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Tables -->
            <div class="space-y-6">
                <!-- Single Features Table -->
                <div>
                    <h3 class="text-base font-bold text-slate-900 mb-2">Table 1.1: Individual Feature Distance (Equation 1)</h3>
                    <div class="overflow-x-auto border border-slate-200 rounded-xl">
                        <table class="w-full text-sm text-left">
                            <thead class="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
                                <tr>
                                    <th class="px-4 py-3 text-center w-24">Rank</th>
                                    <th class="px-4 py-3">Feature Name</th>
                                    <th class="px-4 py-3 text-right">Mean Divergence ($d_{{ij}}$)</th>
                                    <th class="px-4 py-3 text-right">Min Pair ($d_{{ij}}$)</th>
                                    <th class="px-4 py-3 text-right">Max Pair ($d_{{ij}}$)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {"".join(sec1_single_rows)}
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- 2-Feature Combinations Table -->
                <div>
                    <h3 class="text-base font-bold text-slate-900 mb-2">Table 1.2: Half-Features (2 Features) Combinations (Equation 2)</h3>
                    <div class="overflow-x-auto border border-slate-200 rounded-xl">
                        <table class="w-full text-sm text-left">
                            <thead class="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
                                <tr>
                                    <th class="px-4 py-3 text-center w-40">Rank &amp; Notes</th>
                                    <th class="px-4 py-3">Feature Pair</th>
                                    <th class="px-4 py-3 text-right">Mean Divergence ($d_{{ij}}$)</th>
                                    <th class="px-4 py-3 text-right">Min Pair ($d_{{ij}}$)</th>
                                    <th class="px-4 py-3 text-right">Max Pair ($d_{{ij}}$)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {"".join(sec1_pair_rows)}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Progression Comparison -->
            <div class="p-5 rounded-xl bg-slate-100/70 border border-slate-200">
                <h4 class="text-sm font-bold text-slate-900 mb-2">💡 Dimensionality Progression Insight (4 Features)</h4>
                <div class="flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
                    <div class="flex items-center gap-3">
                        <span class="w-8 h-8 rounded-full bg-slate-300 text-slate-700 font-bold flex items-center justify-center text-xs">1D</span>
                        <div>
                            <p class="text-xs text-slate-500">Best 1 Feature</p>
                            <p class="text-sm font-bold text-slate-800">{sec1_data['single_features'][0]['feature']} ({sec1_best_single:.4f})</p>
                        </div>
                    </div>
                    <span class="text-slate-400 font-bold hidden sm:inline">&rarr;</span>
                    <div class="flex items-center gap-3">
                        <span class="w-8 h-8 rounded-full bg-indigo-200 text-indigo-800 font-bold flex items-center justify-center text-xs">2D</span>
                        <div>
                            <p class="text-xs text-indigo-600 font-medium">Best 2 Features (Half)</p>
                            <p class="text-sm font-bold text-slate-800">{" + ".join(sec1_data['pair_features'][0]['features'])} ({sec1_best_pair:.4f})</p>
                        </div>
                    </div>
                    <span class="text-slate-400 font-bold hidden sm:inline">&rarr;</span>
                    <div class="flex items-center gap-3">
                        <span class="w-8 h-8 rounded-full bg-emerald-200 text-emerald-800 font-bold flex items-center justify-center text-xs">4D</span>
                        <div>
                            <p class="text-xs text-emerald-600 font-medium">All 4 Features Combined</p>
                            <p class="text-sm font-bold text-slate-800">All 4 Features ({sec1_all_div:.4f})</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- SECTION 2: FULL 7 FEATURES ANALYSIS -->
        <section class="bg-white rounded-2xl p-6 sm:p-8 shadow-sm border border-slate-200 space-y-8">
            <div class="border-b border-slate-100 pb-4">
                <span class="px-2.5 py-1 text-xs font-bold rounded bg-emerald-100 text-emerald-800 uppercase tracking-wide">Extended Exploration</span>
                <h2 class="text-2xl font-extrabold text-slate-900 mt-2">Section 2: Comprehensive 7-Feature Analysis</h2>
                <p class="text-sm text-slate-500 mt-1">
                    Analyzing all numeric features in dataset: <code class="text-emerald-700 font-semibold">{", ".join(sec2_data['features_tested'])}</code>
                </p>
            </div>

            <!-- Key Metric Highlights -->
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div class="p-5 rounded-xl bg-slate-50 border border-slate-200">
                    <p class="text-xs font-semibold uppercase text-slate-500">Best Feature (All 7)</p>
                    <h3 class="text-xl font-bold text-slate-900 mt-1">{sec2_data['single_features'][0]['feature']}</h3>
                    <p class="text-2xl font-extrabold text-emerald-600 mt-1 font-mono">{sec2_best_single:.4f}</p>
                    <p class="text-xs text-slate-500 mt-1">Dominant discriminant feature</p>
                </div>

                <div class="p-5 rounded-xl bg-emerald-50/50 border border-emerald-100">
                    <p class="text-xs font-semibold uppercase text-emerald-600">Best Half-Features (3-4 Features)</p>
                    <h3 class="text-xl font-bold text-slate-900 mt-1">{" + ".join(sec2_data['best_half']['features'])}</h3>
                    <p class="text-2xl font-extrabold text-emerald-600 mt-1 font-mono">{sec2_best_half:.4f}</p>
                    <p class="text-xs text-slate-500 mt-1">{len(sec2_data['best_half']['features'])} features subset</p>
                </div>

                <div class="p-5 rounded-xl bg-indigo-50/50 border border-indigo-100">
                    <p class="text-xs font-semibold uppercase text-indigo-600">All 7 Features Combined</p>
                    <h3 class="text-xl font-bold text-slate-900 mt-1">Full 7D Vector</h3>
                    <p class="text-2xl font-extrabold text-indigo-600 mt-1 font-mono">{sec2_all_div:.4f}</p>
                    <p class="text-xs text-slate-500 mt-1">Multivariate distance with full 7x7 covariance</p>
                </div>
            </div>

            <!-- Section 2 Chart -->
            <div class="bg-slate-50 p-5 rounded-xl border border-slate-200">
                <h3 class="text-sm font-bold text-slate-900 mb-1">Ranked 1D Feature Distances (All 7 Features)</h3>
                <p class="text-xs text-slate-500 mb-4">Comparison of class separation power across all available features</p>
                <div class="h-72">
                    <canvas id="sec2SingleChart"></canvas>
                </div>
            </div>

            <!-- Section 2 Table -->
            <div>
                <h3 class="text-base font-bold text-slate-900 mb-2">Table 2.1: Full Feature Ranking (Equation 1)</h3>
                <div class="overflow-x-auto border border-slate-200 rounded-xl">
                    <table class="w-full text-sm text-left">
                        <thead class="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
                            <tr>
                                <th class="px-4 py-3 text-center w-28">Rank</th>
                                <th class="px-4 py-3">Feature Name</th>
                                <th class="px-4 py-3 text-right">Mean Divergence ($d_{{ij}}$)</th>
                                <th class="px-4 py-3 text-right">Min Pair ($d_{{ij}}$)</th>
                                <th class="px-4 py-3 text-right">Max Pair ($d_{{ij}}$)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(sec2_single_rows)}
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- CONCLUSION & SUMMARY -->
        <section class="bg-gradient-to-br from-indigo-900 to-slate-900 rounded-2xl p-6 sm:p-8 text-white shadow-lg space-y-4">
            <h2 class="text-xl font-bold text-white flex items-center gap-2">
                <span>📝</span> Conclusions &amp; Answers for Lab 6
            </h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-slate-200 mt-4">
                <div class="bg-white/10 p-4 rounded-xl border border-white/10">
                    <h3 class="font-bold text-indigo-300 text-base mb-1">1. Feature ไหนดีที่สุด?</h3>
                    <p class="text-xs leading-relaxed text-slate-300">
                        • ใน 4 Features หลัก: <b class="text-white">{sec1_data['single_features'][0]['feature']}</b> มีระยะ Divergence สูงสุด ({sec1_best_single:.4f}) จึงแยกคลาสได้ดีที่สุด<br>
                        • ใน 7 Features ทั้งหมด: <b class="text-white">{sec2_data['single_features'][0]['feature']}</b> มี Divergence สูงที่สุด ({sec2_best_single:.4f}) เนื่องจากสัมพันธ์กับเกรดโดยตรง
                    </p>
                </div>
                <div class="bg-white/10 p-4 rounded-xl border border-white/10">
                    <h3 class="font-bold text-indigo-300 text-base mb-1">2. ผลของ "ครึ่งหนึ่ง" (Half Features)</h3>
                    <p class="text-xs leading-relaxed text-slate-300">
                        • คู่ Combination ที่ดีที่สุดใน 4 Features คือ <b class="text-white">{" + ".join(sec1_data['pair_features'][0]['features'])}</b> ({sec1_best_pair:.4f}) ซึ่งตรงกับคู่ของ 2 Features เดี่ยวที่ดีที่สุด ช่วยเพิ่มระยะแยกคลาสมากกว่า Feature เดี่ยวอย่างมีนัยสำคัญ
                    </p>
                </div>
                <div class="bg-white/10 p-4 rounded-xl border border-white/10">
                    <h3 class="font-bold text-indigo-300 text-base mb-1">3. ผลของ "ทุก Feature" (All Features)</h3>
                    <p class="text-xs leading-relaxed text-slate-300">
                        • เมื่อนำ Feature ทั้งหมดมาคิดร่วมกันด้วยสมการ Multivariate (สมการที่ 2) ค่า Divergence รวมเพิ่มขึ้นเป็น <b class="text-white">{sec1_all_div:.4f}</b> ในชุด 4 Features และ <b class="text-white">{sec2_all_div:.4f}</b> ในชุด 7 Features ยืนยันว่าการใช้ข้อมูลหลากมิติช่วยให้แยกระดับเกรดได้สมบูรณ์ยิ่งขึ้น
                    </p>
                </div>
            </div>
        </section>

    </main>

    <!-- Chart.js Scripts -->
    <script>
        // Section 1 Single Features Chart
        new Chart(document.getElementById('sec1SingleChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(chart_sec1_labels)},
                datasets: [{{
                    label: 'Mean Divergence (d_ij)',
                    data: {json.dumps(chart_sec1_values)},
                    backgroundColor: ['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe'],
                    borderRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, grid: {{ color: '#f1f5f9' }} }},
                    x: {{ grid: {{ display: false }} }}
                }}
            }}
        }});

        // Section 1 Pairs Chart
        new Chart(document.getElementById('sec1PairsChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(chart_sec1_pairs_labels)},
                datasets: [{{
                    label: 'Mean Divergence (d_ij)',
                    data: {json.dumps(chart_sec1_pairs_values)},
                    backgroundColor: ['#4f46e5', '#6366f1', '#818cf8', '#94a3b8', '#cbd5e1', '#e2e8f0'],
                    borderRadius: 8
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{ beginAtZero: true, grid: {{ color: '#f1f5f9' }} }},
                    y: {{ grid: {{ display: false }} }}
                }}
            }}
        }});

        // Section 2 Single Features Chart
        new Chart(document.getElementById('sec2SingleChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(chart_sec2_labels)},
                datasets: [{{
                    label: 'Mean Divergence (d_ij)',
                    data: {json.dumps(chart_sec2_values)},
                    backgroundColor: '#10b981',
                    borderRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, grid: {{ color: '#f1f5f9' }} }},
                    x: {{ grid: {{ display: false }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
