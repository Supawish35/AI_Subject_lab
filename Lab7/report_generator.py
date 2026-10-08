import os
import json
import html
from datetime import datetime

def generate_lab7_html_report(summary_data, elbow_data, cluster_stats, pca_data, output_path=None):
    """
    Generate an interactive HTML report for Lab 7 K-Means Clustering.
    Features:
    - Executive summary metrics cards
    - KaTeX formulas for K-Means objective & silhouette calculation
    - Interactive Chart.js graphs for Elbow Method, Silhouette Scores, and PCA 2D Cluster Projection
    - Comprehensive cluster profiling table across all N features
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, "lab7_report.html")

    features = summary_data.get('features', [])
    k_selected = summary_data.get('k_selected', 3)
    dataset_name = summary_data.get('dataset_name', 'Dataset')
    total_samples = summary_data.get('total_samples', 0)
    inertia = summary_data.get('inertia', 0.0)
    silhouette = summary_data.get('silhouette', 0.0)
    iterations = summary_data.get('iterations', 0)
    pca_var_ratio = summary_data.get('pca_explained_variance', [0.0, 0.0])

    # Table rows for cluster profiling
    cluster_rows = []
    palette_classes = [
        ("bg-blue-500", "text-blue-700", "bg-blue-50 border-blue-200"),
        ("bg-emerald-500", "text-emerald-700", "bg-emerald-50 border-emerald-200"),
        ("bg-purple-500", "text-purple-700", "bg-purple-50 border-purple-200"),
        ("bg-amber-500", "text-amber-700", "bg-amber-50 border-amber-200"),
        ("bg-rose-500", "text-rose-700", "bg-rose-50 border-rose-200"),
        ("bg-cyan-500", "text-cyan-700", "bg-cyan-50 border-cyan-200"),
        ("bg-indigo-500", "text-indigo-700", "bg-indigo-50 border-indigo-200"),
        ("bg-orange-500", "text-orange-700", "bg-orange-50 border-orange-200")
    ]

    for c_id, stats in enumerate(cluster_stats):
        color_dot, color_txt, color_box = palette_classes[c_id % len(palette_classes)]
        count = stats['count']
        pct = stats['pct']
        centroid_vals = stats['centroid']
        
        feature_cells = "".join([
            f'<td class="px-4 py-3 text-right font-mono text-slate-700">{centroid_vals.get(f, 0.0):.4f}</td>'
            for f in features
        ])

        cluster_rows.append(f"""
        <tr class="hover:bg-slate-50 transition-colors border-b border-slate-100">
            <td class="px-4 py-3 font-semibold flex items-center gap-2">
                <span class="w-3 h-3 rounded-full {color_dot} inline-block"></span>
                <span>Cluster {c_id}</span>
            </td>
            <td class="px-4 py-3 text-center font-mono font-medium text-slate-800">{count:,} ({pct:.1f}%)</td>
            {feature_cells}
        </tr>
        """)

    feature_headers = "".join([
        f'<th class="px-4 py-3 text-right text-xs font-bold text-slate-600 uppercase tracking-wider">{html.escape(f)}</th>'
        for f in features
    ])

    html_content = f"""<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lab 7: K-Means Clustering Analysis Report</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- KaTeX for Math -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" 
            onload="renderMathInElement(document.body);"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Sarabun:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        body {{
            font-family: 'Plus Jakarta Sans', 'Sarabun', sans-serif;
            background-color: #f8fafc;
        }}
        .font-mono {{
            font-family: 'JetBrains Mono', monospace;
        }}
        .glass-card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(226, 232, 240, 0.8);
        }}
    </style>
</head>
<body class="text-slate-800 antialiased p-4 md:p-8">
    <div class="max-w-7xl mx-auto space-y-8">
        
        <!-- Header Section -->
        <header class="glass-card rounded-2xl p-6 md:p-8 shadow-sm border border-slate-200/80 bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white relative overflow-hidden">
            <div class="absolute -right-10 -bottom-10 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none"></div>
            <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 relative z-10">
                <div>
                    <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/20 text-indigo-300 text-xs font-semibold uppercase tracking-wider mb-2 border border-indigo-400/30">
                        Artificial Intelligence Lab 7
                    </div>
                    <h1 class="text-2xl md:text-3xl font-extrabold tracking-tight">K-Means Clustering Analysis</h1>
                    <p class="text-slate-400 text-sm mt-1">การจัดกลุ่มข้อมูลหลายมิติ (N-Features) และการประเมินประสิทธิภาพคลัสเตอร์</p>
                </div>
                <div class="text-left md:text-right text-xs text-slate-400 space-y-1">
                    <div><span class="text-slate-500">Dataset:</span> <span class="font-mono text-indigo-200">{html.escape(dataset_name)}</span></div>
                    <div><span class="text-slate-500">Generated:</span> {timestamp}</div>
                    <div><span class="text-slate-500">Method:</span> Scikit-Learn KMeans ($k$-means++)</div>
                </div>
            </div>
        </header>

        <!-- Executive Metrics KPI Cards -->
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <div class="glass-card rounded-xl p-4 shadow-sm border-l-4 border-l-indigo-500">
                <span class="text-xs font-semibold text-slate-500 uppercase">Selected K</span>
                <div class="text-2xl font-black text-indigo-700 mt-1">{k_selected} <span class="text-xs font-normal text-slate-400">clusters</span></div>
                <span class="text-xs text-slate-500 mt-1 block">Optimal Cluster Size</span>
            </div>

            <div class="glass-card rounded-xl p-4 shadow-sm border-l-4 border-l-emerald-500">
                <span class="text-xs font-semibold text-slate-500 uppercase">Features Used</span>
                <div class="text-2xl font-black text-emerald-700 mt-1">{len(features)} <span class="text-xs font-normal text-slate-400">dims</span></div>
                <span class="text-xs text-slate-500 mt-1 block">N-Dimensional Data</span>
            </div>

            <div class="glass-card rounded-xl p-4 shadow-sm border-l-4 border-l-blue-500">
                <span class="text-xs font-semibold text-slate-500 uppercase">Total Samples</span>
                <div class="text-2xl font-black text-blue-700 mt-1">{total_samples:,}</div>
                <span class="text-xs text-slate-500 mt-1 block">Data points clustered</span>
            </div>

            <div class="glass-card rounded-xl p-4 shadow-sm border-l-4 border-l-purple-500">
                <span class="text-xs font-semibold text-slate-500 uppercase">Silhouette Score</span>
                <div class="text-2xl font-black text-purple-700 mt-1">{silhouette:.4f}</div>
                <span class="text-xs text-slate-500 mt-1 block">[-1 to 1] Cohesion index</span>
            </div>

            <div class="glass-card rounded-xl p-4 shadow-sm border-l-4 border-l-amber-500">
                <span class="text-xs font-semibold text-slate-500 uppercase">Inertia (WCSS)</span>
                <div class="text-2xl font-black text-amber-700 mt-1">{inertia:,.2f}</div>
                <span class="text-xs text-slate-500 mt-1 block">Within-cluster sum of squares</span>
            </div>

            <div class="glass-card rounded-xl p-4 shadow-sm border-l-4 border-l-cyan-500">
                <span class="text-xs font-semibold text-slate-500 uppercase">Iterations</span>
                <div class="text-2xl font-black text-cyan-700 mt-1">{iterations}</div>
                <span class="text-xs text-slate-500 mt-1 block">Steps to convergence</span>
            </div>
        </div>

        <!-- Math & Formulation Card -->
        <div class="glass-card rounded-xl p-6 shadow-sm border border-slate-200">
            <h3 class="text-base font-bold text-slate-800 mb-3 flex items-center gap-2">
                <svg class="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"/></svg>
                หลักการและสมการคณิตศาสตร์ของ K-Means Clustering (N-Features)
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-slate-600">
                <div class="p-4 bg-slate-50 rounded-lg border border-slate-200">
                    <p class="font-semibold text-slate-700 mb-1">1. Objective Function (WCSS / Inertia):</p>
                    <div class="py-2 text-center text-slate-800">
                        $$J = \\sum_{{k=1}}^{{K}} \\sum_{{\\mathbf{{x}} \\in C_k}} \\|\\mathbf{{x}} - \\boldsymbol{{\\mu}}_k\\|^2$$
                    </div>
                    <p class="text-xs text-slate-500">ลดผลรวมของระยะทางยกกำลังสองจากจุดข้อมูลทุกมิติไปยัง Centroid ในกลุ่ม</p>
                </div>
                <div class="p-4 bg-slate-50 rounded-lg border border-slate-200">
                    <p class="font-semibold text-slate-700 mb-1">2. Centroid Update Rule:</p>
                    <div class="py-2 text-center text-slate-800">
                        $$\\boldsymbol{{\\mu}}_k = \\frac{{1}}{{|C_k|}} \\sum_{{\\mathbf{{x}} \\in C_k}} \\mathbf{{x}}$$
                    </div>
                    <p class="text-xs text-slate-500">ปรับจุดศูนย์กลางด้วยค่าเฉลี่ยของทุก Feature สำหรับทุกข้อมูลที่ตกในคลัสเตอร์</p>
                </div>
                <div class="p-4 bg-slate-50 rounded-lg border border-slate-200">
                    <p class="font-semibold text-slate-700 mb-1">3. Silhouette Coefficient:</p>
                    <div class="py-2 text-center text-slate-800">
                        $$s(i) = \\frac{{b(i) - a(i)}}{{\\max(a(i), b(i))}}$$
                    </div>
                    <p class="text-xs text-slate-500">วัดความแนบแน่นภายในกลุ่ม $a(i)$ เทียบกับความห่างจากกลุ่มที่ใกล้ที่สุด $b(i)$</p>
                </div>
            </div>
        </div>

        <!-- Visualizations Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Elbow Method & Silhouette Chart -->
            <div class="glass-card rounded-xl p-6 shadow-sm border border-slate-200 flex flex-col justify-between">
                <div>
                    <div class="flex justify-between items-center mb-4">
                        <h3 class="font-bold text-slate-800 text-lg">Elbow Curve & Silhouette Score</h3>
                        <span class="text-xs bg-indigo-50 text-indigo-700 px-2 py-1 rounded font-medium border border-indigo-200">Optimal K Evaluation</span>
                    </div>
                    <p class="text-xs text-slate-500 mb-4">
                        กราฟเปรียบเทียบค่า Inertia (WCSS) และ Silhouette Score ในแต่ละค่า $K$ เพื่อหาจุดตัดที่มีเสถียรภาพสูงสุด
                    </p>
                    <div class="h-72">
                        <canvas id="elbowChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- PCA 2D Cluster Space Scatter Plot -->
            <div class="glass-card rounded-xl p-6 shadow-sm border border-slate-200 flex flex-col justify-between">
                <div>
                    <div class="flex justify-between items-center mb-4">
                        <h3 class="font-bold text-slate-800 text-lg">PCA 2D Cluster Projection</h3>
                        <span class="text-xs bg-emerald-50 text-emerald-700 px-2 py-1 rounded font-medium border border-emerald-200">
                            PC1 ({pca_var_ratio[0]*100:.1f}%) + PC2 ({pca_var_ratio[1]*100:.1f}%)
                        </span>
                    </div>
                    <p class="text-xs text-slate-500 mb-4">
                        การลดมิติข้อมูลจาก {len(features)} Features สู่ 2 มิติ (Principal Components) พร้อมแสดงตำแหน่ง Centroids
                    </p>
                    <div class="h-72">
                        <canvas id="pcaChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- Cluster Radar Breakdown Chart -->
        <div class="glass-card rounded-xl p-6 shadow-sm border border-slate-200">
            <div class="flex justify-between items-center mb-4">
                <h3 class="font-bold text-slate-800 text-lg">Cluster Profile Radar Comparison (N-Features)</h3>
                <span class="text-xs bg-purple-50 text-purple-700 px-2 py-1 rounded font-medium border border-purple-200">Normalized Centroid Means</span>
            </div>
            <p class="text-xs text-slate-500 mb-4">
                เปรียบเทียบค่าเฉลี่ยของทุก Feature แต่ละคลัสเตอร์ เพื่อจำแนกลักษณะเฉพาะ (Fingerprint) ของผู้เรียนแต่ละกลุ่ม
            </p>
            <div class="h-80 max-w-2xl mx-auto">
                <canvas id="radarChart"></canvas>
            </div>
        </div>

        <!-- Cluster Details Table -->
        <div class="glass-card rounded-xl p-6 shadow-sm border border-slate-200 overflow-hidden">
            <h3 class="font-bold text-slate-800 text-lg mb-2">ตารางสรุป Centroid และลักษณะของแต่ละ Cluster ({k_selected} Clusters)</h3>
            <p class="text-xs text-slate-500 mb-4">ค่าเฉลี่ยของแต่ละ Feature ตามตำแหน่ง Centroid ศูนย์กลางของแต่ละคลัสเตอร์</p>
            
            <div class="overflow-x-auto rounded-lg border border-slate-200">
                <table class="w-full text-left text-sm">
                    <thead class="bg-slate-100/80 border-b border-slate-200">
                        <tr>
                            <th class="px-4 py-3 text-xs font-bold text-slate-600 uppercase tracking-wider">Cluster</th>
                            <th class="px-4 py-3 text-center text-xs font-bold text-slate-600 uppercase tracking-wider">Size (% of Total)</th>
                            {feature_headers}
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-100 bg-white">
                        {"".join(cluster_rows)}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Feature List Used -->
        <div class="glass-card rounded-xl p-6 shadow-sm border border-slate-200">
            <h4 class="font-bold text-slate-700 text-sm mb-3">รายชื่อ Features ที่นำมาประมวลผล ({len(features)} Features):</h4>
            <div class="flex flex-wrap gap-2">
                {"".join([f'<span class="px-3 py-1 bg-indigo-50 border border-indigo-200 text-indigo-700 text-xs font-mono font-medium rounded-lg">{html.escape(f)}</span>' for f in features])}
            </div>
        </div>

        <!-- Footer -->
        <footer class="text-center text-xs text-slate-400 py-6">
            <p>Artificial Intelligence & Data Mining Laboratory • K-Means N-Dimensional Clustering Analysis</p>
        </footer>

    </div>

    <!-- Chart Configuration Script -->
    <script>
        const elbowData = {json.dumps(elbow_data)};
        const pcaData = {json.dumps(pca_data)};
        const clusterStats = {json.dumps(cluster_stats)};
        const features = {json.dumps(features)};

        // 1. Elbow Chart (Dual Axis: Inertia & Silhouette)
        const ctxElbow = document.getElementById('elbowChart').getContext('2d');
        new Chart(ctxElbow, {{
            type: 'line',
            data: {{
                labels: elbowData.k_values,
                datasets: [
                    {{
                        label: 'Inertia (WCSS)',
                        data: elbowData.inertias,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        yAxisID: 'yInertia',
                        tension: 0.3,
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        borderWidth: 2
                    }},
                    {{
                        label: 'Silhouette Score',
                        data: elbowData.silhouettes,
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        yAxisID: 'ySilhouette',
                        tension: 0.3,
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        borderWidth: 2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Number of Clusters (K)' }} }},
                    yInertia: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{ display: true, text: 'Inertia (WCSS)' }}
                    }},
                    ySilhouette: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{ display: true, text: 'Silhouette Score' }},
                        grid: {{ drawOnChartArea: false }}
                    }}
                }}
            }}
        }});

        // 2. PCA Scatter Plot
        const clusterColors = [
            'rgba(59, 130, 246, 0.6)',
            'rgba(16, 185, 129, 0.6)',
            'rgba(168, 85, 247, 0.6)',
            'rgba(245, 158, 11, 0.6)',
            'rgba(244, 63, 94, 0.6)',
            'rgba(6, 182, 212, 0.6)',
            'rgba(99, 102, 241, 0.6)',
            'rgba(249, 115, 22, 0.6)'
        ];

        const pcaDatasets = [];
        const numClusters = clusterStats.length;

        for (let i = 0; i < numClusters; i++) {{
            const pts = pcaData.points.filter(p => p.cluster === i).map(p => ({{ x: p.x, y: p.y }}));
            pcaDatasets.push({{
                label: `Cluster ${{i}}`,
                data: pts,
                backgroundColor: clusterColors[i % clusterColors.length],
                pointRadius: 3,
                pointHoverRadius: 5
            }});
        }}

        // Add Centroids
        pcaDatasets.push({{
            label: 'Centroids (X)',
            data: pcaData.centroids.map(c => ({{ x: c.x, y: c.y }})),
            backgroundColor: '#0f172a',
            borderColor: '#ffffff',
            borderWidth: 2,
            pointStyle: 'crossRot',
            pointRadius: 9,
            pointHoverRadius: 11
        }});

        const ctxPca = document.getElementById('pcaChart').getContext('2d');
        new Chart(ctxPca, {{
            type: 'scatter',
            data: {{ datasets: pcaDatasets }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Principal Component 1 (PC1)' }} }},
                    y: {{ title: {{ display: true, text: 'Principal Component 2 (PC2)' }} }}
                }}
            }}
        }});

        // 3. Radar Chart for Centroid Profiles
        const radarDatasets = clusterStats.map((cs, idx) => ({{
            label: `Cluster ${{idx}}`,
            data: features.map(f => cs.centroid[f] || 0),
            borderColor: clusterColors[idx % clusterColors.length].replace('0.6', '1'),
            backgroundColor: clusterColors[idx % clusterColors.length].replace('0.6', '0.15'),
            pointRadius: 4,
            borderWidth: 2
        }}));

        const ctxRadar = document.getElementById('radarChart').getContext('2d');
        new Chart(ctxRadar, {{
            type: 'radar',
            data: {{
                labels: features,
                datasets: radarDatasets
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    r: {{
                        beginAtZero: true,
                        pointLabels: {{
                            font: {{ size: 11, family: 'JetBrains Mono' }}
                        }}
                    }}
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
