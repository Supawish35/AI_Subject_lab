import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def build_reactive_website(dataset_path, output_html_path="index.html"):
    """
    Build a standalone, zero-dependency, ultra-reactive K-NN web application.
    Embeds the 10,000-sample dataset, metadata, and an in-browser Euclidean
    K-NN prediction engine with real-time slider reactivity and 2D scatter projection.
    Supports both odd and even K values with distance-weighted proximity tie-breaking.
    """
    df = pd.read_csv(dataset_path)
    target_col = 'FinalGrade' if 'FinalGrade' in df.columns else df.columns[-1]

    drop_cols = [c for c in ['StudentID', 'Name', 'exam_score', 'placement_status'] if c in df.columns and c != target_col]
    df_clean = df.drop(columns=drop_cols).dropna()

    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}

    # Ensure PreviousGrade is mapped to integers
    if 'PreviousGrade' in df_clean.columns:
        df_clean['PreviousGrade_Code'] = [grade_map.get(str(g).strip(), 0) for g in df_clean['PreviousGrade']]
    else:
        df_clean['PreviousGrade_Code'] = 0

    feature_cols = [
        'study_hours',
        'attendance',
        'sleep_hours',
        'internet_usage',
        'assignments_completed',
        'previous_score',
        'PreviousGrade_Code'
    ]

    feature_meta = [
        {'id': 'study_hours', 'name': 'Daily Study Hours', 'min': 1.0, 'max': 11.0, 'unit': 'hrs/day', 'default': 6.0, 'step': 0.5},
        {'id': 'attendance', 'name': 'Class Attendance', 'min': 40.0, 'max': 100.0, 'unit': '%', 'default': 85.0, 'step': 1.0},
        {'id': 'sleep_hours', 'name': 'Daily Sleep', 'min': 3.0, 'max': 10.0, 'unit': 'hrs/night', 'default': 7.0, 'step': 0.5},
        {'id': 'internet_usage', 'name': 'Internet Usage', 'min': 1.0, 'max': 8.0, 'unit': 'hrs/day', 'default': 3.0, 'step': 0.5},
        {'id': 'assignments_completed', 'name': 'Assignments Done', 'min': 0.0, 'max': 10.0, 'unit': 'assignments', 'default': 8.0, 'step': 1.0},
        {'id': 'previous_score', 'name': 'Previous Exam Score', 'min': 35.0, 'max': 95.0, 'unit': 'points', 'default': 78.0, 'step': 1.0},
        {'id': 'PreviousGrade_Code', 'name': 'Previous Academic Grade', 'type': 'select', 'options': ['A', 'B', 'C', 'D', 'F'], 'default': 'B'}
    ]

    y = df_clean[target_col].values
    classes = sorted(list(np.unique(y)))

    # Pack dataset into compact float list [f1, f2, f3, f4, f5, f6, f7, target_grade_idx]
    dataset_records = []
    for idx, row in df_clean.iterrows():
        dataset_records.append([
            round(float(row['study_hours']), 4),
            round(float(row['attendance']), 4),
            round(float(row['sleep_hours']), 4),
            round(float(row['internet_usage']), 4),
            round(float(row['assignments_completed']), 4),
            round(float(row['previous_score']), 4),
            int(row['PreviousGrade_Code']),
            str(row[target_col])
        ])

    dataset_json = json.dumps(dataset_records)
    feature_meta_json = json.dumps(feature_meta)
    classes_json = json.dumps(classes)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lab 4: Reactive K-NN Model Interactive Explorer</title>
    <!-- Tailwind CSS v3 CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    fontFamily: {{
                        sans: ['"Inter"', 'system-ui', 'sans-serif'],
                        mono: ['"JetBrains Mono"', 'monospace']
                    }},
                    colors: {{
                        brand: {{
                            50: '#eef2ff',
                            100: '#e0e7ff',
                            500: '#6366f1',
                            600: '#4f46e5',
                            700: '#4338ca'
                        }}
                    }}
                }}
            }}
        }}
    </script>
    <style>
        body {{
            font-family: 'Inter', system-ui, sans-serif;
            background: #f8fafc;
            color: #0f172a;
        }}
        input[type="range"] {{
            accent-color: #4f46e5;
        }}
        .grade-badge-A {{ background: #10b981; color: white; }}
        .grade-badge-B {{ background: #3b82f6; color: white; }}
        .grade-badge-C {{ background: #f59e0b; color: white; }}
        .grade-badge-D {{ background: #f97316; color: white; }}
        .grade-badge-F {{ background: #ef4444; color: white; }}
    </style>
</head>
<body class="min-h-screen flex flex-col">

    <!-- Navbar -->
    <header class="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-sm">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center gap-3">
                <span class="w-9 h-9 rounded-xl bg-indigo-600 text-white font-black flex items-center justify-center text-lg shadow-md shadow-indigo-600/30">
                    K
                </span>
                <div>
                    <h1 class="text-base font-bold text-slate-900 leading-none">Lab 4: Reactive K-NN Classifier</h1>
                    <span class="text-xs text-slate-500">Real-Time Client-Side Prediction Engine • 10,000 Live Samples</span>
                </div>
            </div>
            <div class="flex items-center gap-3 text-xs font-mono">
                <span class="hidden sm:inline-flex items-center gap-1.5 px-3 py-1 bg-emerald-50 text-emerald-700 rounded-full border border-emerald-200 font-semibold">
                    <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                    10,000 Live Samples
                </span>
                <span id="latencyBadge" class="px-2.5 py-1 bg-slate-100 text-slate-700 rounded-md border border-slate-200">
                    Latency: ~0ms
                </span>
            </div>
        </div>
    </header>

    <!-- Main App Container -->
    <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

        <!-- Top Banner / Preset Profiles -->
        <div class="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
                <h2 class="text-sm font-bold uppercase tracking-wider text-indigo-600">Interactive Model Simulator</h2>
                <p class="text-xs text-slate-500 mt-0.5">Move sliders or select preset student profiles to see instantaneous K-NN Euclidean calculations and majority voting.</p>
            </div>
            <div class="flex flex-wrap items-center gap-2">
                <span class="text-xs font-semibold text-slate-500 mr-1">Load Presets:</span>
                <button onclick="applyPreset('top')" class="px-3 py-1.5 text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-lg hover:bg-emerald-100 transition shadow-sm">
                    ★ Top Student (Grade A)
                </button>
                <button onclick="applyPreset('avg')" class="px-3 py-1.5 text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200 rounded-lg hover:bg-amber-100 transition shadow-sm">
                    ⚡ Average Student (Grade C)
                </button>
                <button onclick="applyPreset('risk')" class="px-3 py-1.5 text-xs font-medium bg-rose-50 text-rose-700 border border-rose-200 rounded-lg hover:bg-rose-100 transition shadow-sm">
                    ⚠ At-Risk Student (Grade D/F)
                </button>
                <button onclick="randomizeInputs()" class="px-3 py-1.5 text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200 rounded-lg hover:bg-slate-200 transition">
                    🎲 Randomize
                </button>
            </div>
        </div>

        <!-- 3-Column Grid: [Sliders Input] | [Prediction Result & Voting] | [Scatter 2D & Neighbors] -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

            <!-- Left Column: Reactive Feature Sliders (4 cols) -->
            <div class="lg:col-span-4 bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4 flex flex-col justify-between">
                <div>
                    <div class="flex items-center justify-between pb-3 border-b border-slate-100">
                        <h3 class="text-sm font-bold text-slate-800 uppercase tracking-wide">Feature Inputs</h3>
                        <span class="text-xs font-mono text-indigo-600 font-semibold bg-indigo-50 px-2 py-0.5 rounded">
                            Reactive
                        </span>
                    </div>

                    <!-- K Value Slider (Supports both ODD and EVEN values: step=1) -->
                    <div class="mt-4 p-3 bg-indigo-50/60 border border-indigo-100 rounded-xl space-y-1.5">
                        <div class="flex justify-between items-center text-xs">
                            <span class="font-bold text-indigo-900">K Neighbors (Odd / Even)</span>
                            <span id="kValBadge" class="font-mono font-black text-indigo-700 text-sm bg-white px-2 py-0.5 rounded border border-indigo-200">K = 5 (Odd)</span>
                        </div>
                        <input id="slider_k" type="range" min="1" max="50" step="1" value="5" oninput="updateK(this.value)" class="w-full cursor-pointer h-2 bg-indigo-200 rounded-lg">
                        <div class="flex justify-between text-[10px] text-indigo-500 font-mono">
                            <span>1 (Strict)</span>
                            <span id="kTypeNote">Even or Odd allowed</span>
                            <span>50 (Broad)</span>
                        </div>
                    </div>

                    <!-- Feature Sliders List -->
                    <div class="space-y-3.5 mt-4" id="slidersContainer">
                        <!-- Injected via JS -->
                    </div>
                </div>

                <div class="pt-3 border-t border-slate-100 text-[11px] text-slate-500 text-center font-mono">
                    Euclidean distance computed dynamically against 10,000 vector records.
                </div>
            </div>

            <!-- Middle Column: Live Prediction Winner & Votes (4 cols) -->
            <div class="lg:col-span-4 space-y-6">

                <!-- Primary Winner Banner Card -->
                <div class="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm text-center relative overflow-hidden">
                    <span class="text-xs font-bold uppercase tracking-wider text-slate-400 block">Predicted Final Grade</span>
                    
                    <div class="my-4 flex items-center justify-center">
                        <div id="predGradeBadge" class="w-24 h-24 rounded-2xl flex items-center justify-center font-black text-5xl shadow-xl transition-all transform duration-300 scale-100">
                            -
                        </div>
                    </div>

                    <div id="confidenceText" class="text-xs font-medium text-slate-600 font-mono">
                        Calculating...
                    </div>

                    <div class="mt-4 pt-3 border-t border-slate-100 flex items-center justify-around text-xs font-mono text-slate-500">
                        <div>Avg Distance: <span id="avgDistVal" class="font-bold text-slate-800">0.0000</span></div>
                        <div>Min Distance: <span id="minDistVal" class="font-bold text-emerald-600">0.0000</span></div>
                    </div>
                </div>

                <!-- Majority Voting Breakdown Bar Chart -->
                <div class="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-3">
                    <div class="flex items-center justify-between pb-2 border-b border-slate-100">
                        <h4 class="text-xs font-bold text-slate-700 uppercase tracking-wide">Neighbor Vote Breakdown</h4>
                        <span id="voteTotalBadge" class="text-xs font-mono text-slate-500">K = 5 Votes</span>
                    </div>

                    <div id="voteBarsList" class="space-y-2 pt-1">
                        <!-- Injected via JS -->
                    </div>
                </div>

                <!-- Distance Metric Selector -->
                <div class="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm text-xs space-y-2">
                    <span class="font-bold text-slate-700 block">Distance Metric & Even-K Tie Breaking</span>
                    <div class="font-mono text-[11px] bg-slate-50 p-2.5 rounded border border-slate-200 text-slate-700">
                        d(x, y) = √( ∑ᵢ₌₁ⁿ (xᵢ - yᵢ)² )
                    </div>
                    <p class="text-[11px] text-slate-500 leading-relaxed">
                        Features are Min-Max scaled into [0.0, 1.0]. When <strong>K is even</strong> and a vote tie occurs (e.g. 2 votes for A and 2 votes for B), ties are resolved by checking which candidate class has the closest nearest neighbor (proximity priority).
                    </p>
                </div>
            </div>

            <!-- Right Column: Nearest Neighbors Table & Feature Space Plot (4 cols) -->
            <div class="lg:col-span-4 bg-white border border-slate-200 rounded-2xl p-5 shadow-sm flex flex-col justify-between space-y-4">
                <div>
                    <div class="flex items-center justify-between pb-3 border-b border-slate-100">
                        <h4 class="text-sm font-bold text-slate-800 uppercase tracking-wide">Nearest Neighbors</h4>
                        <span class="text-xs font-mono text-slate-500">Top Ranked</span>
                    </div>

                    <div class="overflow-y-auto max-h-[300px] mt-3 border border-slate-100 rounded-xl">
                        <table class="w-full text-left text-xs border-collapse">
                            <thead class="bg-slate-50 text-slate-500 sticky top-0 font-semibold border-b border-slate-200">
                                <tr>
                                    <th class="py-2 px-2.5">Rank</th>
                                    <th class="py-2 px-2.5">ID</th>
                                    <th class="py-2 px-2.5 text-right">Distance</th>
                                    <th class="py-2 px-2.5 text-center">Class</th>
                                </tr>
                            </thead>
                            <tbody id="neighborsTableBody" class="divide-y divide-slate-100 font-mono text-[11px]">
                                <!-- Injected via JS -->
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- 2D Projection Chart (Study Hours vs Previous Score) -->
                <div>
                    <div class="flex items-center justify-between pb-1 border-b border-slate-100 mb-2">
                        <h5 class="text-xs font-bold text-slate-700 uppercase">2D Space: Study Hours vs Score</h5>
                        <span class="text-[10px] text-indigo-600 font-mono">Current + Top Neighbors</span>
                    </div>
                    <div class="h-44 relative">
                        <canvas id="projectionChart"></canvas>
                    </div>
                </div>
            </div>

        </div>

    </main>

    <!-- Footer -->
    <footer class="bg-white border-t border-slate-200 mt-8 py-4 text-center text-xs text-slate-400 font-mono">
        Lab 4 • K-Nearest Neighbors Reactive Simulator • 10,000 Records
    </footer>

    <!-- Core Reactive Script -->
    <script>
        const RAW_DATA = {dataset_json}; // [f1, f2, f3, f4, f5, f6, f7, target_grade]
        const META = {feature_meta_json};
        const CLASSES = {classes_json};
        
        let CURRENT_K = 5;
        let chartInstance = null;

        const userInputs = {{
            'study_hours': 6.0,
            'attendance': 85.0,
            'sleep_hours': 7.0,
            'internet_usage': 3.0,
            'assignments_completed': 8.0,
            'previous_score': 78.0,
            'PreviousGrade_Code': 1 // Grade 'B'
        }};

        function initSliders() {{
            const container = document.getElementById('slidersContainer');
            container.innerHTML = '';

            META.forEach(f => {{
                if (f.id === 'PreviousGrade_Code') {{
                    const div = document.createElement('div');
                    div.className = 'space-y-1 text-xs';
                    const gradeLetters = ['A', 'B', 'C', 'D', 'F'];
                    const currentLetter = gradeLetters[userInputs[f.id]] || 'B';
                    div.innerHTML = `
                        <div class="flex justify-between font-semibold text-slate-700">
                            <span>${{f.name}}</span>
                            <span id="badge_${{f.id}}" class="font-mono text-indigo-600 font-bold">${{currentLetter}}</span>
                        </div>
                        <select onchange="updateFeature('${{f.id}}', this.value)" class="w-full text-xs py-1.5 px-2 bg-slate-50 border border-slate-200 rounded-lg font-mono focus:ring-1 focus:ring-indigo-500">
                            <option value="0" ${{userInputs[f.id] === 0 ? 'selected' : ''}}>A</option>
                            <option value="1" ${{userInputs[f.id] === 1 ? 'selected' : ''}}>B</option>
                            <option value="2" ${{userInputs[f.id] === 2 ? 'selected' : ''}}>C</option>
                            <option value="3" ${{userInputs[f.id] === 3 ? 'selected' : ''}}>D</option>
                            <option value="4" ${{userInputs[f.id] === 4 ? 'selected' : ''}}>F</option>
                        </select>
                    `;
                    container.appendChild(div);
                }} else {{
                    const div = document.createElement('div');
                    div.className = 'space-y-1 text-xs';
                    div.innerHTML = `
                        <div class="flex justify-between items-center font-medium text-slate-700">
                            <span>${{f.name}}</span>
                            <span id="badge_${{f.id}}" class="font-mono font-bold text-indigo-600 bg-slate-100 px-1.5 py-0.5 rounded text-[11px]">
                                ${{userInputs[f.id]}} ${{f.unit}}
                            </span>
                        </div>
                        <input type="range" min="${{f.min}}" max="${{f.max}}" step="${{f.step}}" value="${{userInputs[f.id]}}"
                            oninput="updateFeature('${{f.id}}', this.value, '${{f.unit}}')"
                            class="w-full cursor-pointer h-1.5 bg-slate-200 rounded-lg">
                        <div class="flex justify-between text-[10px] text-slate-400 font-mono">
                            <span>${{f.min}}</span>
                            <span>${{f.max}} ${{f.unit}}</span>
                        </div>
                    `;
                    container.appendChild(div);
                }}
            }});
        }}

        function updateFeature(id, val, unit) {{
            userInputs[id] = parseFloat(val);
            const badge = document.getElementById('badge_' + id);
            if (badge) {{
                if (id === 'PreviousGrade_Code') {{
                    const names = ['A', 'B', 'C', 'D', 'F'];
                    badge.innerText = names[parseInt(val)] || 'A';
                }} else {{
                    badge.innerText = val + ' ' + (unit || '');
                }}
            }}
            runKnnPrediction();
        }}

        function updateK(val) {{
            CURRENT_K = parseInt(val);
            const isEven = (CURRENT_K % 2 === 0);
            const parityLabel = isEven ? 'Even' : 'Odd';
            document.getElementById('kValBadge').innerText = `K = ${{CURRENT_K}} (${{parityLabel}})`;
            document.getElementById('voteTotalBadge').innerText = `K = ${{CURRENT_K}} Votes`;
            runKnnPrediction();
        }}

        function normalizeVector(inputs) {{
            const vec = [];
            META.forEach(f => {{
                if (f.id === 'PreviousGrade_Code') {{
                    vec.push(inputs[f.id]);
                }} else {{
                    const min = f.min;
                    const max = f.max;
                    const val = inputs[f.id];
                    const norm = Math.max(0, Math.min(1, (val - min) / (max - min)));
                    vec.push(norm);
                }}
            }});
            return vec;
        }}

        function runKnnPrediction() {{
            const t0 = performance.now();
            const targetVec = normalizeVector(userInputs);

            const nFeatures = 7;
            const distances = [];

            for (let i = 0; i < RAW_DATA.length; i++) {{
                const row = RAW_DATA[i];
                let sumSq = 0;
                for (let f = 0; f < nFeatures; f++) {{
                    const diff = targetVec[f] - row[f];
                    sumSq += diff * diff;
                }}
                distances.push({{
                    id: i,
                    distance: Math.sqrt(sumSq),
                    grade: row[7],
                    study_hours: row[0],
                    previous_score: row[5]
                }});
            }}

            distances.sort((a, b) => a.distance - b.distance);
            const topK = distances.slice(0, CURRENT_K);

            const votes = {{ 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0 }};
            topK.forEach(item => {{
                votes[item.grade] = (votes[item.grade] || 0) + 1;
            }});

            // Find candidates with highest votes
            let maxVotes = -1;
            let tiedCandidates = [];
            for (const g of CLASSES) {{
                const count = votes[g] || 0;
                if (count > maxVotes) {{
                    maxVotes = count;
                    tiedCandidates = [g];
                }} else if (count === maxVotes && maxVotes > 0) {{
                    tiedCandidates.push(g);
                }}
            }}

            // Tie-break resolution (especially useful when K is even)
            let winningGrade = tiedCandidates[0] || 'A';
            let tieBroken = false;
            if (tiedCandidates.length > 1) {{
                tieBroken = true;
                let bestDistance = Infinity;
                for (const cand of tiedCandidates) {{
                    // Find distance to the closest neighbor belonging to this candidate class
                    const candItem = topK.find(item => item.grade === cand);
                    if (candItem && candItem.distance < bestDistance) {{
                        bestDistance = candItem.distance;
                        winningGrade = cand;
                    }}
                }}
            }}

            const t1 = performance.now();
            document.getElementById('latencyBadge').innerText = 'Latency: ' + (t1 - t0).toFixed(1) + 'ms';

            renderResults(winningGrade, maxVotes, votes, topK, targetVec, tieBroken);
        }}

        function renderResults(winner, maxVotes, votes, topK, targetVec, tieBroken) {{
            const badge = document.getElementById('predGradeBadge');
            badge.innerText = winner;
            badge.className = `w-24 h-24 rounded-2xl flex items-center justify-center font-black text-5xl shadow-xl transition-all transform duration-200 grade-badge-${{winner}}`;

            const winPct = ((maxVotes / CURRENT_K) * 100).toFixed(1);
            const tieNote = tieBroken ? ' (Tie resolved by nearest proximity)' : '';
            document.getElementById('confidenceText').innerText = `${{maxVotes}} of ${{CURRENT_K}} neighbors voted for Grade ${{winner}} (${{winPct}}%)${{tieNote}}`;

            const avgDist = (topK.reduce((acc, cur) => acc + cur.distance, 0) / topK.length).toFixed(4);
            const minDist = topK[0].distance.toFixed(4);
            document.getElementById('avgDistVal').innerText = avgDist;
            document.getElementById('minDistVal').innerText = minDist;

            const voteList = document.getElementById('voteBarsList');
            voteList.innerHTML = '';
            CLASSES.forEach(c => {{
                const cnt = votes[c] || 0;
                const pct = ((cnt / CURRENT_K) * 100).toFixed(0);
                const isWinner = (c === winner);
                const barColor = isWinner ? 'bg-indigo-600' : 'bg-slate-300';
                const textColor = isWinner ? 'font-bold text-indigo-700' : 'text-slate-600';

                const row = document.createElement('div');
                row.className = 'flex items-center gap-2 text-xs';
                row.innerHTML = `
                    <span class="w-7 font-bold text-slate-700">Class ${{c}}</span>
                    <div class="flex-1 bg-slate-100 rounded-full h-2.5 overflow-hidden">
                        <div class="${{barColor}} h-full rounded-full transition-all duration-200" style="width: ${{pct}}%"></div>
                    </div>
                    <span class="w-16 text-right font-mono ${{textColor}}">${{cnt}}/${{CURRENT_K}} (${{pct}}%)</span>
                `;
                voteList.appendChild(row);
            }});

            const tbody = document.getElementById('neighborsTableBody');
            tbody.innerHTML = '';
            topK.forEach((n, idx) => {{
                const tr = document.createElement('tr');
                tr.className = 'hover:bg-slate-50 transition-colors';
                tr.innerHTML = `
                    <td class="py-1.5 px-2.5 text-slate-400">#${{idx + 1}}</td>
                    <td class="py-1.5 px-2.5 text-slate-600 font-mono">ID #${{n.id}}</td>
                    <td class="py-1.5 px-2.5 text-right font-semibold text-indigo-600">${{n.distance.toFixed(4)}}</td>
                    <td class="py-1.5 px-2.5 text-center">
                        <span class="px-2 py-0.5 rounded text-[10px] font-bold grade-badge-${{n.grade}}">
                            ${{n.grade}}
                        </span>
                    </td>
                `;
                tbody.appendChild(tr);
            }});

            updateChart(targetVec, topK);
        }}

        function updateChart(targetVec, topK) {{
            const ctx = document.getElementById('projectionChart').getContext('2d');

            const neighborPoints = topK.map(n => ({{
                x: n.study_hours,
                y: n.previous_score
            }}));

            const currentPoint = [{{
                x: targetVec[0],
                y: targetVec[5]
            }}];

            if (chartInstance) {{
                chartInstance.data.datasets[0].data = currentPoint;
                chartInstance.data.datasets[1].data = neighborPoints;
                chartInstance.update();
            }} else {{
                chartInstance = new Chart(ctx, {{
                    type: 'scatter',
                    data: {{
                        datasets: [
                            {{
                                label: 'Query Student',
                                data: currentPoint,
                                backgroundColor: '#4f46e5',
                                borderColor: '#312e81',
                                pointRadius: 8,
                                pointHoverRadius: 10
                            }},
                            {{
                                label: 'K Neighbors',
                                data: neighborPoints,
                                backgroundColor: '#10b981',
                                borderColor: '#065f46',
                                pointRadius: 5,
                                pointHoverRadius: 7
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                position: 'top',
                                labels: {{ boxWidth: 10, font: {{ size: 10 }} }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                min: 0,
                                max: 1,
                                title: {{ display: true, text: 'Study Hours (Norm)', font: {{ size: 10 }} }}
                            }},
                            y: {{
                                min: 0,
                                max: 1,
                                title: {{ display: true, text: 'Score (Norm)', font: {{ size: 10 }} }}
                            }}
                        }}
                    }}
                }});
            }}
        }}

        function applyPreset(type) {{
            if (type === 'top') {{
                userInputs.study_hours = 10.0;
                userInputs.attendance = 96.0;
                userInputs.sleep_hours = 8.0;
                userInputs.internet_usage = 2.0;
                userInputs.assignments_completed = 10.0;
                userInputs.previous_score = 92.0;
                userInputs.PreviousGrade_Code = 0;
            }} else if (type === 'avg') {{
                userInputs.study_hours = 5.0;
                userInputs.attendance = 75.0;
                userInputs.sleep_hours = 7.0;
                userInputs.internet_usage = 4.0;
                userInputs.assignments_completed = 6.0;
                userInputs.previous_score = 68.0;
                userInputs.PreviousGrade_Code = 2;
            }} else if (type === 'risk') {{
                userInputs.study_hours = 1.5;
                userInputs.attendance = 45.0;
                userInputs.sleep_hours = 4.0;
                userInputs.internet_usage = 7.0;
                userInputs.assignments_completed = 2.0;
                userInputs.previous_score = 40.0;
                userInputs.PreviousGrade_Code = 4;
            }}
            initSliders();
            runKnnPrediction();
        }}

        function randomizeInputs() {{
            META.forEach(f => {{
                if (f.id === 'PreviousGrade_Code') {{
                    userInputs[f.id] = Math.floor(Math.random() * 5);
                }} else {{
                    const range = f.max - f.min;
                    const rand = f.min + Math.random() * range;
                    userInputs[f.id] = parseFloat(rand.toFixed(1));
                }}
            }});
            initSliders();
            runKnnPrediction();
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            initSliders();
            runKnnPrediction();
        }});
    </script>
</body>
</html>
"""
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"[✓] Reactive K-NN Web App successfully compiled to: {output_html_path}")
    return output_html_path

if __name__ == '__main__':
    dataset = '../DataSet/student_dataset_full_normalized.csv'
    build_reactive_website(dataset, 'index.html')
