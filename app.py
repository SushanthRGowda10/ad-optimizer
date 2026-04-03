"""
Smart Ad Optimization and Campaign Analytics Platform
Main Flask Application Entry Point
"""

import os
import pandas as pd
from flask import Flask, redirect, url_for
from flask_login import LoginManager, current_user

from config import Config
from models import db, User, Dataset


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # ✅ Create required folders
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

    # ✅ Initialize DB
    db.init_app(app)

    # ✅ Login setup
    login_manager = LoginManager(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # ✅ Global storage for trained models
    app.trained_models = {}

    # ✅ Inject dataset status into templates
    @app.context_processor
    def inject_has_data():
        if current_user.is_authenticated:
            ds = Dataset.query.filter_by(
                user_id=current_user.id,
                is_active=True
            ).first()
            return {'has_data': ds is not None}
        return {'has_data': False}

    # ✅ Import ALL blueprints
    from routes.auth_routes import auth_bp
    from routes.dataset_routes import dataset_bp
    from routes.analytics_routes import analytics_bp
    from routes.prediction_routes import prediction_bp
    
    

    # ✅ Register ALL blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(dataset_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(prediction_bp)
    
            # 🔥 IMPORTANT

    # ✅ Create DB tables
    with app.app_context():
        db.create_all()

    return app


# ✅ Create app instance
app = create_app()

@app.route('/ab-test', methods=['GET', 'POST'])
def ab_test():
    from flask import request

    ctr_a = ""
    ctr_b = ""
    result = ""
    winner = ""

    if request.method == 'POST':
        print("POST RECEIVED")
        ctr_a = request.form.get('ctr_a', '').strip()
        ctr_b = request.form.get('ctr_b', '').strip()

        try:
            a = float(ctr_a)
            b = float(ctr_b)

            if a > b:
                result = f"Ad A performs better ({a}% vs {b}%)."
                winner = "Ad A"
            elif b > a:
                result = f"Ad B performs better ({b}% vs {a}%)."
                winner = "Ad B"
            else:
                result = f"Both ads perform equally ({a}%)."
                winner = "Tie"
        except ValueError:
            result = "Invalid input. Enter numbers only."

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>A/B Testing — AdOptimizer</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}

            :root {{
                --bg-base: #0a0f1e;
                --bg-surface: #111827;
                --bg-card: #1a2235;
                --bg-card-2: #0f172a;
                --border: rgba(255,255,255,0.08);
                --text-primary: #f1f5f9;
                --text-secondary: #94a3b8;
                --accent: #6366f1;
                --accent-2: #06b6d4;
                --success: #10b981;
                --warning: #f59e0b;
                --danger: #ef4444;
                --sidebar-w: 240px;
                --topbar-h: 60px;
                --radius: 16px;
                --shadow: 0 18px 40px rgba(0,0,0,0.28);
            }}

            body {{
                font-family: Arial, sans-serif;
                background: linear-gradient(180deg, #081224 0%, #0a0f1e 100%);
                color: var(--text-primary);
                min-height: 100vh;
                overflow-x: hidden;
            }}

            .sidebar {{
                position: fixed;
                top: 0;
                left: 0;
                width: var(--sidebar-w);
                height: 100vh;
                background: var(--bg-surface);
                border-right: 1px solid var(--border);
                z-index: 100;
            }}

            .sidebar-header {{
                padding: 22px 18px;
                border-bottom: 1px solid var(--border);
                font-size: 26px;
                font-weight: 700;
            }}

            .sidebar-header span {{
                font-size: 20px;
                margin-left: 8px;
                vertical-align: middle;
            }}

            .menu {{
                padding: 16px 0;
            }}

            .menu-title {{
                padding: 14px 18px 8px;
                font-size: 11px;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }}

            .menu a {{
                display: block;
                text-decoration: none;
                color: var(--text-secondary);
                padding: 11px 18px;
                font-size: 15px;
                border-left: 3px solid transparent;
                transition: 0.2s;
            }}

            .menu a:hover,
            .menu a.active {{
                color: var(--accent);
                background: rgba(99,102,241,0.12);
                border-left-color: var(--accent);
            }}

            .main {{
                margin-left: var(--sidebar-w);
                min-height: 100vh;
            }}

            .topbar {{
                height: var(--topbar-h);
                background: rgba(17,24,39,0.9);
                border-bottom: 1px solid var(--border);
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 24px;
                position: sticky;
                top: 0;
                backdrop-filter: blur(10px);
                z-index: 50;
            }}

            .topbar-title {{
                font-size: 20px;
                font-weight: 700;
            }}

            .topbar-badge {{
                font-size: 13px;
                color: var(--text-secondary);
                padding: 8px 12px;
                border-radius: 999px;
                background: rgba(255,255,255,0.05);
                border: 1px solid var(--border);
            }}

            .content {{
                padding: 24px;
            }}

            .hero {{
                position: relative;
                overflow: hidden;
                background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(6,182,212,0.10));
                border: 1px solid rgba(99,102,241,0.26);
                border-radius: var(--radius);
                padding: 28px;
                margin-bottom: 24px;
                box-shadow: var(--shadow);
            }}

            .hero::after {{
                content: "";
                position: absolute;
                right: -80px;
                top: -80px;
                width: 220px;
                height: 220px;
                background: radial-gradient(circle, rgba(99,102,241,0.22), transparent 70%);
                border-radius: 50%;
            }}

            .hero h1 {{
                font-size: 34px;
                margin-bottom: 8px;
                position: relative;
                z-index: 1;
            }}

            .hero p {{
                color: var(--text-secondary);
                font-size: 15px;
                line-height: 1.7;
                position: relative;
                z-index: 1;
            }}

            .hero-tags {{
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin-top: 16px;
                position: relative;
                z-index: 1;
            }}

            .hero-tag {{
                padding: 8px 12px;
                border-radius: 999px;
                font-size: 13px;
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.08);
                color: var(--text-secondary);
            }}

            .grid {{
                display: grid;
                grid-template-columns: 380px 1fr;
                gap: 22px;
                margin-bottom: 22px;
            }}

            .card {{
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 24px;
                box-shadow: var(--shadow);
            }}

            .card-header {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 10px;
            }}

            .card-icon {{
                width: 42px;
                height: 42px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, var(--accent), var(--accent-2));
                color: white;
                font-size: 20px;
                font-weight: 700;
            }}

            .card h3 {{
                font-size: 22px;
            }}

            .sub {{
                color: var(--text-secondary);
                line-height: 1.7;
                margin-bottom: 20px;
                font-size: 14px;
            }}

            .form-group {{
                margin-bottom: 18px;
            }}

            label {{
                display: block;
                font-size: 12px;
                font-weight: 700;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 8px;
            }}

            .input-wrap {{
                position: relative;
            }}

            .input-prefix {{
                position: absolute;
                left: 14px;
                top: 50%;
                transform: translateY(-50%);
                color: var(--text-secondary);
                font-size: 14px;
            }}

            input {{
                width: 100%;
                padding: 15px 16px 15px 38px;
                border-radius: 12px;
                border: 1px solid var(--border);
                background: var(--bg-card-2);
                color: var(--text-primary);
                font-size: 18px;
                outline: none;
                transition: 0.2s;
            }}

            input:focus {{
                border-color: var(--accent);
                box-shadow: 0 0 0 3px rgba(99,102,241,0.14);
            }}

            .btn {{
                width: 100%;
                padding: 15px 18px;
                border: none;
                border-radius: 12px;
                background: linear-gradient(135deg, var(--accent), var(--accent-2));
                color: white;
                font-size: 16px;
                font-weight: 700;
                cursor: pointer;
                transition: 0.18s;
                box-shadow: 0 12px 24px rgba(99,102,241,0.22);
            }}

            .btn:hover {{
                transform: translateY(-1px);
                opacity: 0.96;
            }}

            .result-wrap {{
                min-height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
            }}

            .result-icon {{
                width: 88px;
                height: 88px;
                border-radius: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 40px;
                margin-bottom: 18px;
                background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(6,182,212,0.18));
                border: 1px solid rgba(99,102,241,0.18);
            }}

            .result-title {{
                font-size: 30px;
                font-weight: 700;
                margin-bottom: 10px;
            }}

            .result-text {{
                font-size: 17px;
                color: #dbe4ef;
                line-height: 1.8;
                margin-bottom: 18px;
                max-width: 620px;
            }}

            .badge {{
                display: inline-block;
                padding: 11px 18px;
                border-radius: 999px;
                font-weight: 700;
                margin-bottom: 20px;
                font-size: 14px;
            }}

            .badge-a {{
                color: var(--success);
                background: rgba(16,185,129,0.14);
                border: 1px solid rgba(16,185,129,0.28);
            }}

            .badge-b {{
                color: var(--accent-2);
                background: rgba(6,182,212,0.14);
                border: 1px solid rgba(6,182,212,0.28);
            }}

            .badge-tie {{
                color: var(--warning);
                background: rgba(245,158,11,0.14);
                border: 1px solid rgba(245,158,11,0.28);
            }}

            .stats {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 16px;
                width: 100%;
                margin-top: 6px;
            }}

            .stat {{
                background: var(--bg-card-2);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 18px;
                text-align: center;
            }}

            .stat-label {{
                color: var(--text-secondary);
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 8px;
            }}

            .stat-value {{
                font-size: 30px;
                font-weight: 700;
            }}

            .insight-row {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 16px;
                margin-bottom: 22px;
            }}

            .insight {{
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 18px;
                box-shadow: var(--shadow);
            }}

            .insight small {{
                display: block;
                color: var(--text-secondary);
                text-transform: uppercase;
                font-size: 11px;
                letter-spacing: 0.06em;
                margin-bottom: 8px;
            }}

            .insight strong {{
                font-size: 22px;
            }}

            .chart-card {{
                background: var(--bg-card);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 24px;
                box-shadow: var(--shadow);
            }}

            .chart-card-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 14px;
            }}

            .chart-card-header h3 {{
                font-size: 22px;
            }}

            .chart-note {{
                color: var(--text-secondary);
                font-size: 13px;
            }}

            .chart-box {{
                height: 350px;
            }}

            .footer-note {{
                margin-top: 18px;
                color: #64748b;
                font-size: 13px;
                text-align: center;
            }}

            @media (max-width: 1100px) {{
                .grid {{
                    grid-template-columns: 1fr;
                }}

                .insight-row {{
                    grid-template-columns: 1fr;
                }}
            }}

            @media (max-width: 920px) {{
                .sidebar {{
                    display: none;
                }}

                .main {{
                    margin-left: 0;
                }}

                .stats {{
                    grid-template-columns: 1fr;
                }}

                .hero h1 {{
                    font-size: 28px;
                }}

                .result-title {{
                    font-size: 24px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="sidebar">
            <div class="sidebar-header">📊 <span>AdOptimizer</span></div>

            <div class="menu">
                <div class="menu-title">Main</div>
                <a href="/dashboard">Dashboard</a>
                <a href="/analytics">Analytics</a>
                <a href="/prediction">AI Prediction</a>
                <a href="/optimization">Optimization</a>
                <a href="/reports">Reports</a>

                <div class="menu-title">Tools</div>
                <a href="/ab-test" class="active">A/B Testing</a>
            </div>
        </div>

        <div class="main">
            <div class="topbar">
                <div class="topbar-title">A/B Testing Simulator</div>
                <div class="topbar-badge">CTR-based comparison</div>
            </div>

            <div class="content">
                <div class="hero">
                    <h1>Compare Ad Variants Smarter</h1>
                    <p>
                        Test two ad creatives side by side using CTR values, identify the stronger performer instantly,
                        and visualize the difference using a clean comparison chart.
                    </p>
                    <div class="hero-tags">
                        <div class="hero-tag">Fast comparison</div>
                        <div class="hero-tag">Visual winner highlight</div>
                        <div class="hero-tag">Chart-based analysis</div>
                    </div>
                </div>

                <div class="insight-row">
                    <div class="insight">
                        <small>Variant A</small>
                        <strong>{ctr_a + "%" if ctr_a else "—"}</strong>
                    </div>
                    <div class="insight">
                        <small>Variant B</small>
                        <strong>{ctr_b + "%" if ctr_b else "—"}</strong>
                    </div>
                    <div class="insight">
                        <small>Winner</small>
                        <strong>{winner if winner else "Waiting"}</strong>
                    </div>
                </div>

                <div class="grid">
                    <div class="card">
                        <div class="card-header">
                            <div class="card-icon">⚖️</div>
                            <h3>Compare Ads</h3>
                        </div>
                        <p class="sub">
                            Enter the CTR values for Ad A and Ad B. The simulator will evaluate both and show the winning variation clearly.
                        </p>

                        <form method="POST">
                            <div class="form-group">
                                <label>CTR of Ad A (%)</label>
                                <div class="input-wrap">
                                    <div class="input-prefix">A</div>
                                    <input type="number" step="0.01" name="ctr_a" value="{ctr_a}" required>
                                </div>
                            </div>

                            <div class="form-group">
                                <label>CTR of Ad B (%)</label>
                                <div class="input-wrap">
                                    <div class="input-prefix">B</div>
                                    <input type="number" step="0.01" name="ctr_b" value="{ctr_b}" required>
                                </div>
                            </div>

                            <button type="submit" class="btn">Compare Ads</button>
                        </form>
                    </div>

                    <div class="card">
                        <div class="result-wrap">
                            {
                                f'''
                                <div class="result-icon">📈</div>
                                <div class="result-title">Comparison Result</div>
                                <div class="result-text">{result}</div>
                                <div class="badge {"badge-a" if winner == "Ad A" else "badge-b" if winner == "Ad B" else "badge-tie"}">
                                    Winner: {winner}
                                </div>

                                <div class="stats">
                                    <div class="stat">
                                        <div class="stat-label">Ad A CTR</div>
                                        <div class="stat-value">{ctr_a}%</div>
                                    </div>
                                    <div class="stat">
                                        <div class="stat-label">Ad B CTR</div>
                                        <div class="stat-value">{ctr_b}%</div>
                                    </div>
                                </div>
                                '''
                                if result else
                                '''
                                <div class="result-icon">🧪</div>
                                <div class="result-title">Run an A/B Test</div>
                                <div class="result-text">
                                    Add CTR values for both ad variants and click Compare Ads to reveal the winner, summary, and graph.
                                </div>
                                '''
                            }

                            <div class="footer-note">Debug: A={ctr_a}, B={ctr_b}</div>
                        </div>
                    </div>
                </div>

                <div class="chart-card">
                    <div class="chart-card-header">
                        <h3>CTR Comparison Graph</h3>
                        <div class="chart-note">Visual performance comparison of both variants</div>
                    </div>
                    <div class="chart-box">
                        <canvas id="abChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const chartA = {float(ctr_a) if ctr_a else 0};
            const chartB = {float(ctr_b) if ctr_b else 0};

            const ctx = document.getElementById('abChart').getContext('2d');
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: ['Ad A', 'Ad B'],
                    datasets: [{{
                        label: 'CTR (%)',
                        data: [chartA, chartB],
                        backgroundColor: [
                            'rgba(99, 102, 241, 0.78)',
                            'rgba(6, 182, 212, 0.78)'
                        ],
                        borderColor: [
                            'rgba(99, 102, 241, 1)',
                            'rgba(6, 182, 212, 1)'
                        ],
                        borderWidth: 1,
                        borderRadius: 10,
                        borderSkipped: false
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{
                                color: '#f1f5f9'
                            }}
                        }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    return 'CTR: ' + context.parsed.y + '%';
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            ticks: {{
                                color: '#94a3b8'
                            }},
                            grid: {{
                                color: 'rgba(255,255,255,0.06)'
                            }}
                        }},
                        y: {{
                            beginAtZero: true,
                            ticks: {{
                                color: '#94a3b8'
                            }},
                            grid: {{
                                color: 'rgba(255,255,255,0.06)'
                            }}
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """


# ✅ Default route
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('analytics.dashboard'))
    return redirect(url_for('auth.login'))


# ✅ Run server
if __name__ == '__main__':
    app.run(debug=True, port=5001)