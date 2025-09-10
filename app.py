import os
import pandas as pd
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===== Configuration =====
REFRESH_INTERVAL = 30000  # 30 seconds in milliseconds

# Available teams for dropdown
AVAILABLE_TEAMS = ["Group 1", "Group 2", "Group 3", "Group 4"]

# In-memory storage for scores
scores_db = []

# ===== Scoring Functions =====
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
                "ML Model Scoring"
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
        dbc.Col([  # LLM Scoring interface and Analysis scoring interface can be added similarly.
        ], width=4),
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
    return current_time, json.dumps(scores_db, default=str)

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
    
    # Create team summary with all scores
    team_scores = {}
    
    # Initialize teams with default values
    for team in AVAILABLE_TEAMS:
        team_scores[team] = {
            'ml_score': 0, 'llm_score': 0, 'analysis_score': 0, 'total_score': 0,
            'ml_count': 0, 'llm_count': 0, 'analysis_count': 0
        }
    
    # Process scores from in-memory storage
    for score in scores_db:
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
