"""
Hackathon Webapp - UK Housing Market Challenge
Railway Deployment Version
Fetches group metrics from DagsHub and provides scoring interface
"""

import os
import json
import time
import requests
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===== Configuration =====
DAGSHUB_REPO = "smksean/hackathon-logging"
DAGSHUB_BASE_URL = f"https://dagshub.com/api/v1/repos/{DAGSHUB_REPO}"
REFRESH_INTERVAL = 30000  # 30 seconds in milliseconds

# Mock data for development (replace with actual DagsHub API calls)
MOCK_EXPERIMENTS = [
    {
        "id": "exp_001",
        "team_name": "Group 1",
        "rmse": 180268.5,
        "mse": 32496700000.0,
        "r2_score": 0.847,
        "creativity": 8.5,
        "inference_speed": 1.2,
        "timestamp": datetime.now() - timedelta(minutes=5),
        "model_type": "Gradient Boosting",
        "features_used": 15
    },
    {
        "id": "exp_002", 
        "team_name": "Group 2",
        "rmse": 195432.1,
        "mse": 38193700000.0,
        "r2_score": 0.823,
        "creativity": 9.2,
        "inference_speed": 0.8,
        "timestamp": datetime.now() - timedelta(minutes=12),
        "model_type": "Neural Network",
        "features_used": 22
    },
    {
        "id": "exp_003",
        "team_name": "Group 3", 
        "rmse": 165789.3,
        "mse": 27486000000.0,
        "r2_score": 0.891,
        "creativity": 7.8,
        "inference_speed": 2.1,
        "timestamp": datetime.now() - timedelta(minutes=8),
        "model_type": "Random Forest",
        "features_used": 18
    },
    {
        "id": "exp_004",
        "team_name": "Group 4",
        "rmse": 210456.7,
        "mse": 44292000000.0,
        "r2_score": 0.765,
        "creativity": 9.5,
        "inference_speed": 0.6,
        "timestamp": datetime.now() - timedelta(minutes=15),
        "model_type": "XGBoost",
        "features_used": 12
    }
]

# Available teams for dropdown
AVAILABLE_TEAMS = ["Group 1", "Group 2", "Group 3", "Group 4"]

# CSV file for persistent storage
SCORES_CSV = "hackathon_scores.csv"

# Initialize CSV file if it doesn't exist
def init_scores_csv():
    if not os.path.exists(SCORES_CSV):
        # Check if sample data exists
        if os.path.exists('sample_scores.csv'):
            # Copy sample data to main CSV
            import shutil
            shutil.copy('sample_scores.csv', SCORES_CSV)
        else:
            # Create empty CSV with headers
            with open(SCORES_CSV, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'team_name', 'score_type', 'r2_score', 'rmse', 'mse', 
                               'creativity', 'accuracy', 'speed', 'quality', 'completeness', 
                               'innovation', 'model_type', 'notes'])

# Load scores from CSV
def load_scores_from_csv():
    scores = []
    if os.path.exists(SCORES_CSV):
        with open(SCORES_CSV, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                scores.append(row)
    return scores

# Save score to CSV
def save_score_to_csv(score_data):
    with open(SCORES_CSV, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            score_data.get('timestamp', ''),
            score_data.get('team_name', ''),
            score_data.get('score_type', ''),
            score_data.get('r2_score', ''),
            score_data.get('rmse', ''),
            score_data.get('mse', ''),
            score_data.get('creativity', ''),
            score_data.get('accuracy', ''),
            score_data.get('speed', ''),
            score_data.get('quality', ''),
            score_data.get('completeness', ''),
            score_data.get('innovation', ''),
            score_data.get('model_type', ''),
            score_data.get('notes', '')
        ])

# Initialize CSV file
init_scores_csv()

# ===== DagsHub Integration =====
def fetch_dagshub_experiments() -> List[Dict]:
    """
    Fetch experiment data from DagsHub API
    Returns mock data for now - replace with actual API calls
    """
    try:
        # TODO: Replace with actual DagsHub API calls
        # headers = {"Authorization": f"token {os.getenv('DAGSHUB_TOKEN')}"}
        # response = requests.get(f"{DAGSHUB_BASE_URL}/experiments", headers=headers)
        # return response.json()
        
        # For now, return mock data with some randomization
        import random
        experiments = []
        for exp in MOCK_EXPERIMENTS:
            exp_copy = exp.copy()
            # Add some randomness to simulate live updates
            exp_copy["rmse"] += random.uniform(-5000, 5000)
            exp_copy["r2_score"] += random.uniform(-0.01, 0.01)
            exp_copy["creativity"] += random.uniform(-0.2, 0.2)
            exp_copy["inference_speed"] += random.uniform(-0.1, 0.1)
            experiments.append(exp_copy)
        return experiments
    except Exception as e:
        print(f"Error fetching DagsHub data: {e}")
        return MOCK_EXPERIMENTS

def calculate_ml_score(rmse: float, mse: float, r2_score: float) -> float:
    """Calculate ML model performance score (0-100)"""
    # Normalize metrics to 0-100 scale
    r2_normalized = r2_score * 100  # R2 is already 0-1
    rmse_normalized = max(0, 100 - (rmse / 5000))  # Lower RMSE is better
    mse_normalized = max(0, 100 - (mse / 1000000000))  # Lower MSE is better
    return (r2_normalized * 0.5) + (rmse_normalized * 0.3) + (mse_normalized * 0.2)

def calculate_llm_score(creativity: float, accuracy: float, speed: float) -> float:
    """Calculate LLM implementation score (0-100)"""
    # Normalize to 0-100 scale
    creativity_normalized = creativity * 10  # 0-10 to 0-100
    accuracy_normalized = accuracy * 100  # 0-1 to 0-100
    speed_normalized = max(0, 100 - (speed * 10))  # Lower speed is better
    return (creativity_normalized * 0.4) + (accuracy_normalized * 0.4) + (speed_normalized * 0.2)

def calculate_analysis_score(quality: float, completeness: float, innovation: float) -> float:
    """Calculate analysis quality score (0-100)"""
    # All inputs are 0-10, normalize to 0-100
    return (quality * 10 * 0.4) + (completeness * 10 * 0.4) + (innovation * 10 * 0.2)

def calculate_total_score(ml_score: float, llm_score: float, analysis_score: float) -> float:
    """Calculate total composite score (0-100)"""
    # Weighted combination: 50% ML, 30% LLM, 20% Analysis
    return (ml_score * 0.5) + (llm_score * 0.3) + (analysis_score * 0.2)

# ===== Dash App =====
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,  # Professional theme
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
])

app.title = "UK Housing Market Hackathon - Live Leaderboard"

# ===== Layout Components =====
def create_header():
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("UK Housing Market Hackathon", className="mb-0", style={"color": "#2c3e50"}),
                    html.Small("Live Leaderboard & Scoring Interface", className="text-muted")
                ], width=8),
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-clock me-2", style={"color": "#7f8c8d"}),
                        html.Span(id="current-time", className="h5 mb-0", style={"color": "#2c3e50"})
                    ], className="text-end")
                ], width=4)
            ], align="center")
        ], fluid=True),
        color="light",
        className="mb-4 shadow-sm border-bottom"
    )

def create_leaderboard_card():
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-trophy me-2", style={"color": "#f39c12"}),
                "Live Leaderboard"
            ], className="mb-0"),
            html.Small("Updates every 30 seconds", className="text-muted")
        ]),
        dbc.CardBody([
            html.Div(id="leaderboard-content"),
            dcc.Interval(
                id='leaderboard-interval',
                interval=REFRESH_INTERVAL,
                n_intervals=0
            )
        ])
    ], className="shadow-sm")

def create_metrics_overview():
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-chart-line me-2", style={"color": "#3498db"}),
                "Performance Metrics"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-bullseye fa-2x mb-2", style={"color": "#27ae60"}),
                        html.H3(id="avg-total", className="mb-0", style={"color": "#27ae60"}),
                        html.Small("Average Total Score", className="text-muted")
                    ], className="text-center p-3")
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-brain fa-2x mb-2", style={"color": "#3498db"}),
                        html.H3(id="avg-ml", className="mb-0", style={"color": "#3498db"}),
                        html.Small("Average ML Score", className="text-muted")
                    ], className="text-center p-3")
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-robot fa-2x mb-2", style={"color": "#9b59b6"}),
                        html.H3(id="avg-llm", className="mb-0", style={"color": "#9b59b6"}),
                        html.Small("Average LLM Score", className="text-muted")
                    ], className="text-center p-3")
                ], width=4)
            ])
        ])
    ], className="shadow-sm")

def create_ml_scoring_interface():
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-brain me-2", style={"color": "#3498db"}),
                "ML Model Scoring (DagsHub Metrics)"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            # Admin Password Section
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Admin Password"),
                        dbc.Input(id="ml-admin-password", type="password", placeholder="Enter admin password")
                    ], width=6),
                    dbc.Col([
                        dbc.Label(" "),  # Spacer
                        dbc.Button("Authenticate", id="ml-auth-btn", color="primary", className="w-100")
                    ], width=6)
                ], className="mb-3"),
                html.Div(id="ml-auth-status", className="mb-3")
            ], id="ml-auth-section"),
            
            # Scoring Form (initially hidden)
            html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Team Name"),
                    dcc.Dropdown(
                        id="ml-score-team",
                        options=[{"label": team, "value": team} for team in AVAILABLE_TEAMS],
                        placeholder="Select team..."
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Model Type"),
                    dbc.Select(
                        id="ml-score-model-type",
                        options=[
                            {"label": "Gradient Boosting", "value": "gradient_boosting"},
                            {"label": "Random Forest", "value": "random_forest"},
                            {"label": "Neural Network", "value": "neural_network"},
                            {"label": "XGBoost", "value": "xgboost"},
                            {"label": "Other", "value": "other"}
                        ],
                        value="gradient_boosting"
                    )
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("R² Score (0-1)"),
                    dbc.Input(id="ml-score-r2", type="number", min=0, max=1, step=0.001, value=0.8)
                ], width=4),
                dbc.Col([
                    dbc.Label("RMSE"),
                    dbc.Input(id="ml-score-rmse", type="number", min=0, step=1, value=180000)
                ], width=4),
                dbc.Col([
                    dbc.Label("MSE"),
                    dbc.Input(id="ml-score-mse", type="number", min=0, step=1000000, value=32000000000)
                ], width=4)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button("Submit ML Score", id="submit-ml-score-btn", color="primary", size="lg", className="w-100")
                ], width=6),
                dbc.Col([
                    dbc.Button("Clear ML Form", id="clear-ml-score-btn", color="secondary", size="lg", className="w-100")
                ], width=6)
            ]),
            
            html.Div(id="ml-score-feedback", className="mt-3")
            ], id="ml-scoring-form", style={"display": "none"})
        ])
    ], className="shadow-sm")

def create_llm_scoring_interface():
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-robot me-2", style={"color": "#9b59b6"}),
                "LLM Implementation Scoring"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            # Admin Password Section
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Admin Password"),
                        dbc.Input(id="llm-admin-password", type="password", placeholder="Enter admin password")
                    ], width=6),
                    dbc.Col([
                        dbc.Label(" "),  # Spacer
                        dbc.Button("Authenticate", id="llm-auth-btn", color="primary", className="w-100")
                    ], width=6)
                ], className="mb-3"),
                html.Div(id="llm-auth-status", className="mb-3")
            ], id="llm-auth-section"),
            
            # Scoring Form (initially hidden)
            html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Team Name"),
                    dcc.Dropdown(
                        id="llm-score-team",
                        options=[{"label": team, "value": team} for team in AVAILABLE_TEAMS],
                        placeholder="Select team..."
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("LLM Model"),
                    dbc.Select(
                        id="llm-score-model",
                        options=[
                            {"label": "GPT-4", "value": "gpt4"},
                            {"label": "Claude", "value": "claude"},
                            {"label": "Llama", "value": "llama"},
                            {"label": "Custom", "value": "custom"}
                        ],
                        value="gpt4"
                    )
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Creativity Score (0-10)"),
                    dbc.Input(id="llm-score-creativity", type="number", min=0, max=10, step=0.1, value=7.5)
                ], width=4),
                dbc.Col([
                    dbc.Label("Accuracy Score (0-1)"),
                    dbc.Input(id="llm-score-accuracy", type="number", min=0, max=1, step=0.001, value=0.8)
                ], width=4),
                dbc.Col([
                    dbc.Label("Inference Speed (seconds)"),
                    dbc.Input(id="llm-score-speed", type="number", min=0.1, max=10, step=0.1, value=1.5)
                ], width=4)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Additional Notes"),
                    dbc.Textarea(id="llm-score-notes", placeholder="Any additional comments...", rows=3)
                ], width=12)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button("Submit LLM Score", id="submit-llm-score-btn", color="success", size="lg", className="w-100")
                ], width=6),
                dbc.Col([
                    dbc.Button("Clear LLM Form", id="clear-llm-score-btn", color="secondary", size="lg", className="w-100")
                ], width=6)
            ]),
            
            html.Div(id="llm-score-feedback", className="mt-3")
            ], id="llm-scoring-form", style={"display": "none"})
        ])
    ], className="shadow-sm")

def create_analysis_scoring_interface():
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-chart-pie me-2", style={"color": "#e67e22"}),
                "Analysis Quality Scoring"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            # Admin Password Section
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Admin Password"),
                        dbc.Input(id="analysis-admin-password", type="password", placeholder="Enter admin password")
                    ], width=6),
                    dbc.Col([
                        dbc.Label(" "),  # Spacer
                        dbc.Button("Authenticate", id="analysis-auth-btn", color="primary", className="w-100")
                    ], width=6)
                ], className="mb-3"),
                html.Div(id="analysis-auth-status", className="mb-3")
            ], id="analysis-auth-section"),
            
            # Scoring Form (initially hidden)
            html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Team Name"),
                    dcc.Dropdown(
                        id="analysis-score-team",
                        options=[{"label": team, "value": team} for team in AVAILABLE_TEAMS],
                        placeholder="Select team..."
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Analysis Type"),
                    dbc.Select(
                        id="analysis-score-type",
                        options=[
                            {"label": "EDA Analysis", "value": "eda"},
                            {"label": "Feature Engineering", "value": "feature_eng"},
                            {"label": "Model Selection", "value": "model_selection"},
                            {"label": "Business Insights", "value": "business_insights"}
                        ],
                        value="eda"
                    )
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Quality Score (0-10)"),
                    dbc.Input(id="analysis-score-quality", type="number", min=0, max=10, step=0.1, value=7.5)
                ], width=4),
                dbc.Col([
                    dbc.Label("Completeness (0-10)"),
                    dbc.Input(id="analysis-score-completeness", type="number", min=0, max=10, step=0.1, value=8.0)
                ], width=4),
                dbc.Col([
                    dbc.Label("Innovation (0-10)"),
                    dbc.Input(id="analysis-score-innovation", type="number", min=0, max=10, step=0.1, value=7.0)
                ], width=4)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Analysis Notes"),
                    dbc.Textarea(id="analysis-score-notes", placeholder="Detailed feedback on analysis quality...", rows=3)
                ], width=12)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button("Submit Analysis Score", id="submit-analysis-score-btn", color="info", size="lg", className="w-100")
                ], width=6),
                dbc.Col([
                    dbc.Button("Clear Analysis Form", id="clear-analysis-score-btn", color="secondary", size="lg", className="w-100")
                ], width=6)
            ]),
            
            html.Div(id="analysis-score-feedback", className="mt-3")
            ], id="analysis-scoring-form", style={"display": "none"})
        ])
    ], className="shadow-sm")

def create_performance_charts():
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                html.I(className="fas fa-chart-area me-2", style={"color": "#34495e"}),
                "Performance Analytics"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="total-score-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="speed-vs-accuracy-chart")
                ], width=6)
            ])
        ])
    ], className="shadow-sm")

# ===== App Layout =====
app.layout = dbc.Container([
    create_header(),
    
    dbc.Row([
        dbc.Col([
            create_leaderboard_card()
        ], width=8),
        dbc.Col([
            create_metrics_overview()
        ], width=4)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            create_ml_scoring_interface()
        ], width=4),
        dbc.Col([
            create_llm_scoring_interface()
        ], width=4),
        dbc.Col([
            create_analysis_scoring_interface()
        ], width=4)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            create_performance_charts()
        ], width=12)
    ], className="mb-4"),
    
    # Hidden div to store current data
    html.Div(id="current-data", style={"display": "none"})
    
], fluid=True, className="p-4")

# ===== Callbacks =====
@app.callback(
    [Output("current-time", "children"),
     Output("current-data", "children")],
    [Input("leaderboard-interval", "n_intervals")]
)
def update_time_and_data(n_intervals):
    current_time = datetime.now().strftime("%H:%M:%S")
    experiments = fetch_dagshub_experiments()
    return current_time, json.dumps(experiments, default=str)

@app.callback(
    [Output("leaderboard-content", "children"),
     Output("avg-total", "children"),
     Output("avg-ml", "children"), 
     Output("avg-llm", "children")],
    [Input("current-data", "children")]
)
def update_leaderboard(data_json):
    if not data_json:
        return "Loading...", "0.0", "0.0", "0.0"
    
    # Load scores from CSV
    csv_scores = load_scores_from_csv()
    
    # Get team data from DagsHub (mock data for now)
    experiments = json.loads(data_json)
    
    # Create team summary with all scores
    team_scores = {}
    
    # Initialize teams with default values
    for team in AVAILABLE_TEAMS:
        team_scores[team] = {
            'ml_score': 0, 'llm_score': 0, 'analysis_score': 0, 'total_score': 0,
            'ml_count': 0, 'llm_count': 0, 'analysis_count': 0
        }
    
    # Process CSV scores
    for score in csv_scores:
        team = score['team_name']
        score_type = score['score_type']
        
        if team in team_scores:
            if score_type == 'ml':
                r2 = float(score.get('r2_score', 0) or 0)
                rmse = float(score.get('rmse', 0) or 0)
                mse = float(score.get('mse', 0) or 0)
                ml_score = calculate_ml_score(rmse, mse, r2)
                team_scores[team]['ml_score'] += ml_score
                team_scores[team]['ml_count'] += 1
            elif score_type == 'llm':
                creativity = float(score.get('creativity', 0) or 0)
                accuracy = float(score.get('accuracy', 0) or 0)
                speed = float(score.get('speed', 0) or 0)
                llm_score = calculate_llm_score(creativity, accuracy, speed)
                team_scores[team]['llm_score'] += llm_score
                team_scores[team]['llm_count'] += 1
            elif score_type == 'analysis':
                quality = float(score.get('quality', 0) or 0)
                completeness = float(score.get('completeness', 0) or 0)
                innovation = float(score.get('innovation', 0) or 0)
                analysis_score = calculate_analysis_score(quality, completeness, innovation)
                team_scores[team]['analysis_score'] += analysis_score
                team_scores[team]['analysis_count'] += 1
    
    # Calculate averages and total scores
    leaderboard_data = []
    for team, scores in team_scores.items():
        # Calculate averages (show 0 if no scores yet)
        avg_ml = scores['ml_score'] / max(1, scores['ml_count']) if scores['ml_count'] > 0 else 0
        avg_llm = scores['llm_score'] / max(1, scores['llm_count']) if scores['llm_count'] > 0 else 0
        avg_analysis = scores['analysis_score'] / max(1, scores['analysis_count']) if scores['analysis_count'] > 0 else 0
        
        # Calculate total score
        total_score = calculate_total_score(avg_ml, avg_llm, avg_analysis)
        
        leaderboard_data.append({
            'team_name': team,
            'ml_score': avg_ml,
            'llm_score': avg_llm,
            'analysis_score': avg_analysis,
            'total_score': total_score
        })
    
    # Sort by total score
    leaderboard_data.sort(key=lambda x: x['total_score'], reverse=True)
    
    # Create leaderboard table
    leaderboard_rows = []
    for i, team_data in enumerate(leaderboard_data):
        rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        
        row = dbc.Row([
            dbc.Col([
                html.H5(f"{rank_icon} {team_data['team_name']}", 
                       className="mb-1", 
                       style={"color": "#f39c12" if i == 0 else "#2c3e50"})
            ], width=3),
            dbc.Col([
                html.Span(f"{team_data['ml_score']:.1f}", 
                         className="badge me-1", 
                         style={"background-color": "#3498db", "color": "white"}),
                html.Small("ML Score", className="text-muted")
            ], width=2),
            dbc.Col([
                html.Span(f"{team_data['llm_score']:.1f}", 
                         className="badge me-1", 
                         style={"background-color": "#9b59b6", "color": "white"}),
                html.Small("LLM Score", className="text-muted")
            ], width=2),
            dbc.Col([
                html.Span(f"{team_data['analysis_score']:.1f}", 
                         className="badge me-1", 
                         style={"background-color": "#e67e22", "color": "white"}),
                html.Small("Analysis Score", className="text-muted")
            ], width=2),
            dbc.Col([
                html.Span(f"{team_data['total_score']:.1f}", 
                         className="badge me-1", 
                         style={"background-color": "#27ae60", "color": "white"}),
                html.Small("Total Score", className="text-muted")
            ], width=2),
            dbc.Col([
                html.Small(f"ML:{team_scores[team_data['team_name']]['ml_count']} LLM:{team_scores[team_data['team_name']]['llm_count']} A:{team_scores[team_data['team_name']]['analysis_count']}", 
                          className="text-muted")
            ], width=1)
        ], className="mb-2 p-2 border rounded")
        leaderboard_rows.append(row)
    
    # If no leaderboard rows created, show empty teams
    if not leaderboard_rows:
        for i, team in enumerate(AVAILABLE_TEAMS):
            rank_icon = f"{i+1}."
            row = dbc.Row([
                dbc.Col([
                    html.H5(f"{rank_icon} {team}", 
                           className="mb-1", 
                           style={"color": "#2c3e50"})
                ], width=3),
                dbc.Col([
                    html.Span("0.0", 
                             className="badge me-1", 
                             style={"background-color": "#bdc3c7", "color": "white"}),
                    html.Small("ML Score", className="text-muted")
                ], width=2),
                dbc.Col([
                    html.Span("0.0", 
                             className="badge me-1", 
                             style={"background-color": "#bdc3c7", "color": "white"}),
                    html.Small("LLM Score", className="text-muted")
                ], width=2),
                dbc.Col([
                    html.Span("0.0", 
                             className="badge me-1", 
                             style={"background-color": "#bdc3c7", "color": "white"}),
                    html.Small("Analysis Score", className="text-muted")
                ], width=2),
                dbc.Col([
                    html.Span("0.0", 
                             className="badge me-1", 
                             style={"background-color": "#bdc3c7", "color": "white"}),
                    html.Small("Total Score", className="text-muted")
                ], width=2),
                dbc.Col([
                    html.Small("No scores yet", className="text-muted")
                ], width=1)
            ], className="mb-2 p-2 border rounded")
            leaderboard_rows.append(row)
    
    # Calculate overall averages
    if leaderboard_data:
        avg_total = sum(t['total_score'] for t in leaderboard_data) / len(leaderboard_data)
        avg_ml = sum(t['ml_score'] for t in leaderboard_data) / len(leaderboard_data)
        avg_llm = sum(t['llm_score'] for t in leaderboard_data) / len(leaderboard_data)
    else:
        avg_total = avg_ml = avg_llm = 0.0
    
    return leaderboard_rows, f"{avg_total:.1f}", f"{avg_ml:.1f}", f"{avg_llm:.1f}"

@app.callback(
    [Output("total-score-chart", "figure"),
     Output("speed-vs-accuracy-chart", "figure")],
    [Input("current-data", "children")]
)
def update_charts(data_json):
    if not data_json:
        return {}, {}
    
    # Load scores from CSV for charts
    csv_scores = load_scores_from_csv()
    
    if not csv_scores:
        # Return empty charts if no data
        fig1 = px.bar(title="Total Scores by Team")
        fig2 = px.scatter(title="LLM Speed vs Accuracy")
        return fig1, fig2
    
    # Process CSV scores for charts
    team_data = {}
    llm_data = []
    
    for score in csv_scores:
        team = score['team_name']
        score_type = score['score_type']
        
        if team not in team_data:
            team_data[team] = {'ml_scores': [], 'llm_scores': [], 'analysis_scores': []}
        
        if score_type == 'ml':
            r2 = float(score.get('r2_score', 0) or 0)
            rmse = float(score.get('rmse', 0) or 0)
            mse = float(score.get('mse', 0) or 0)
            ml_score = calculate_ml_score(rmse, mse, r2)
            team_data[team]['ml_scores'].append(ml_score)
        elif score_type == 'llm':
            creativity = float(score.get('creativity', 0) or 0)
            accuracy = float(score.get('accuracy', 0) or 0)
            speed = float(score.get('speed', 0) or 0)
            llm_score = calculate_llm_score(creativity, accuracy, speed)
            team_data[team]['llm_scores'].append(llm_score)
            llm_data.append({'team': team, 'speed': speed, 'accuracy': accuracy, 'creativity': creativity})
        elif score_type == 'analysis':
            quality = float(score.get('quality', 0) or 0)
            completeness = float(score.get('completeness', 0) or 0)
            innovation = float(score.get('innovation', 0) or 0)
            analysis_score = calculate_analysis_score(quality, completeness, innovation)
            team_data[team]['analysis_scores'].append(analysis_score)
    
    # Calculate total scores for each team
    total_scores = []
    for team, scores in team_data.items():
        avg_ml = sum(scores['ml_scores']) / max(1, len(scores['ml_scores']))
        avg_llm = sum(scores['llm_scores']) / max(1, len(scores['llm_scores']))
        avg_analysis = sum(scores['analysis_scores']) / max(1, len(scores['analysis_scores']))
        total_score = calculate_total_score(avg_ml, avg_llm, avg_analysis)
        total_scores.append({'team': team, 'total_score': total_score})
    
    # Total Score chart
    if total_scores:
        df_total = pd.DataFrame(total_scores)
        fig1 = px.bar(
            df_total, x="team", y="total_score",
            title="Total Scores by Team",
            color="total_score",
            color_continuous_scale="RdYlGn"
        )
    else:
        fig1 = px.bar(title="Total Scores by Team")
    
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='black'
    )
    
    # Speed vs Accuracy chart (LLM data only)
    if llm_data:
        df_llm = pd.DataFrame(llm_data)
        fig2 = px.scatter(
            df_llm, x="speed", y="accuracy",
            size="creativity", color="team",
            title="LLM Speed vs Accuracy Trade-off",
            hover_data=["creativity"]
        )
    else:
        fig2 = px.scatter(title="LLM Speed vs Accuracy")
    
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='black'
    )
    
    return fig1, fig2

# Admin Authentication Callbacks
@app.callback(
    [Output("ml-auth-status", "children"),
     Output("ml-scoring-form", "style"),
     Output("ml-auth-section", "style")],
    [Input("ml-auth-btn", "n_clicks")],
    [State("ml-admin-password", "value")]
)
def authenticate_ml_admin(n_clicks, password):
    if not n_clicks:
        return "", {"display": "none"}, {}
    
    if password == "1998":
        return dbc.Alert("✅ Authenticated! You can now input scores.", color="success"), {}, {"display": "none"}
    else:
        return dbc.Alert("❌ Invalid password. Access denied.", color="danger"), {"display": "none"}, {}

@app.callback(
    [Output("llm-auth-status", "children"),
     Output("llm-scoring-form", "style"),
     Output("llm-auth-section", "style")],
    [Input("llm-auth-btn", "n_clicks")],
    [State("llm-admin-password", "value")]
)
def authenticate_llm_admin(n_clicks, password):
    if not n_clicks:
        return "", {"display": "none"}, {}
    
    if password == "1998":
        return dbc.Alert("✅ Authenticated! You can now input scores.", color="success"), {}, {"display": "none"}
    else:
        return dbc.Alert("❌ Invalid password. Access denied.", color="danger"), {"display": "none"}, {}

@app.callback(
    [Output("analysis-auth-status", "children"),
     Output("analysis-scoring-form", "style"),
     Output("analysis-auth-section", "style")],
    [Input("analysis-auth-btn", "n_clicks")],
    [State("analysis-admin-password", "value")]
)
def authenticate_analysis_admin(n_clicks, password):
    if not n_clicks:
        return "", {"display": "none"}, {}
    
    if password == "1998":
        return dbc.Alert("✅ Authenticated! You can now input scores.", color="success"), {}, {"display": "none"}
    else:
        return dbc.Alert("❌ Invalid password. Access denied.", color="danger"), {"display": "none"}, {}

# ML Scoring Callback
@app.callback(
    [Output("ml-score-feedback", "children"),
     Output("ml-score-team", "value"),
     Output("ml-score-r2", "value"),
     Output("ml-score-rmse", "value"),
     Output("ml-score-mse", "value")],
    [Input("submit-ml-score-btn", "n_clicks"),
     Input("clear-ml-score-btn", "n_clicks")],
    [State("ml-score-team", "value"),
     State("ml-score-model-type", "value"),
     State("ml-score-r2", "value"),
     State("ml-score-rmse", "value"),
     State("ml-score-mse", "value")]
)
def handle_ml_scoring(submit_clicks, clear_clicks, team_name, model_type, r2, rmse, mse):
    ctx = callback_context
    if not ctx.triggered:
        return "", "", 0.8, 180000, 32000000000
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "clear-ml-score-btn":
        return "", "", 0.8, 180000, 32000000000
    
    elif button_id == "submit-ml-score-btn":
        if not team_name:
            return dbc.Alert("Please select a team!", color="danger"), "", r2, rmse, mse
        
        # Save to CSV
        score_data = {
            'timestamp': datetime.now().isoformat(),
            'team_name': team_name,
            'score_type': 'ml',
            'r2_score': r2,
            'rmse': rmse,
            'mse': mse,
            'model_type': model_type
        }
        save_score_to_csv(score_data)
        
        success_msg = f"ML Score submitted for {team_name}: R²={r2:.3f}, RMSE={rmse:,.0f}, MSE={mse:,.0f}"
        return dbc.Alert(success_msg, color="success"), "", 0.8, 180000, 32000000000
    
    return "", "", r2, rmse, mse

# LLM Scoring Callback
@app.callback(
    [Output("llm-score-feedback", "children"),
     Output("llm-score-team", "value"),
     Output("llm-score-creativity", "value"),
     Output("llm-score-accuracy", "value"),
     Output("llm-score-speed", "value"),
     Output("llm-score-notes", "value")],
    [Input("submit-llm-score-btn", "n_clicks"),
     Input("clear-llm-score-btn", "n_clicks")],
    [State("llm-score-team", "value"),
     State("llm-score-model", "value"),
     State("llm-score-creativity", "value"),
     State("llm-score-accuracy", "value"),
     State("llm-score-speed", "value"),
     State("llm-score-notes", "value")]
)
def handle_llm_scoring(submit_clicks, clear_clicks, team_name, model, creativity, accuracy, speed, notes):
    ctx = callback_context
    if not ctx.triggered:
        return "", "", 7.5, 0.8, 1.5, ""
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "clear-llm-score-btn":
        return "", "", 7.5, 0.8, 1.5, ""
    
    elif button_id == "submit-llm-score-btn":
        if not team_name:
            return dbc.Alert("Please select a team!", color="danger"), "", creativity, accuracy, speed, notes
        
        # Save to CSV
        score_data = {
            'timestamp': datetime.now().isoformat(),
            'team_name': team_name,
            'score_type': 'llm',
            'creativity': creativity,
            'accuracy': accuracy,
            'speed': speed,
            'model_type': model,
            'notes': notes
        }
        save_score_to_csv(score_data)
        
        success_msg = f"LLM Score submitted for {team_name}: Creativity={creativity}, Accuracy={accuracy:.3f}, Speed={speed}s"
        return dbc.Alert(success_msg, color="success"), "", 7.5, 0.8, 1.5, ""
    
    return "", "", creativity, accuracy, speed, notes

# Analysis Scoring Callback
@app.callback(
    [Output("analysis-score-feedback", "children"),
     Output("analysis-score-team", "value"),
     Output("analysis-score-quality", "value"),
     Output("analysis-score-completeness", "value"),
     Output("analysis-score-innovation", "value"),
     Output("analysis-score-notes", "value")],
    [Input("submit-analysis-score-btn", "n_clicks"),
     Input("clear-analysis-score-btn", "n_clicks")],
    [State("analysis-score-team", "value"),
     State("analysis-score-type", "value"),
     State("analysis-score-quality", "value"),
     State("analysis-score-completeness", "value"),
     State("analysis-score-innovation", "value"),
     State("analysis-score-notes", "value")]
)
def handle_analysis_scoring(submit_clicks, clear_clicks, team_name, analysis_type, quality, completeness, innovation, notes):
    ctx = callback_context
    if not ctx.triggered:
        return "", "", 7.5, 8.0, 7.0, ""
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "clear-analysis-score-btn":
        return "", "", 7.5, 8.0, 7.0, ""
    
    elif button_id == "submit-analysis-score-btn":
        if not team_name:
            return dbc.Alert("Please select a team!", color="danger"), "", quality, completeness, innovation, notes
        
        # Save to CSV
        score_data = {
            'timestamp': datetime.now().isoformat(),
            'team_name': team_name,
            'score_type': 'analysis',
            'quality': quality,
            'completeness': completeness,
            'innovation': innovation,
            'model_type': analysis_type,
            'notes': notes
        }
        save_score_to_csv(score_data)
        
        success_msg = f"Analysis Score submitted for {team_name}: Quality={quality}, Completeness={completeness}, Innovation={innovation}"
        return dbc.Alert(success_msg, color="success"), "", 7.5, 8.0, 7.0, ""
    
    return "", "", quality, completeness, innovation, notes

# ===== Server Configuration for Railway =====
server = app.server

if __name__ == "__main__":
    # Railway will set the PORT environment variable
    port = int(os.environ.get("PORT", 8051))
    app.run_server(host="0.0.0.0", port=port, debug=False)
