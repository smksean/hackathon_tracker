"""
Hackathon Webapp - UK Housing Market Challenge
Railway Deployment Version (No CSV)
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd

# -----------------------------------------------------------
# Utility
# -----------------------------------------------------------

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

# -----------------------------------------------------------
# Dash App
# -----------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "UK Housing Hackathon Scoring"

# -----------------------------------------------------------
# Layout
# -----------------------------------------------------------

app.layout = dbc.Container([
    html.H1("🏠 UK Housing Market Hackathon"),
    html.Hr(),

    # Tabs for scoring
    dcc.Tabs(id="tabs", value="ml", children=[
        dcc.Tab(label="ML Scoring", value="ml"),
        dcc.Tab(label="LLM Scoring", value="llm"),
        dcc.Tab(label="Analysis Scoring", value="analysis"),
        dcc.Tab(label="Leaderboard", value="leaderboard"),
    ]),

    html.Div(id="tab-content", className="p-4")
], fluid=True)

# -----------------------------------------------------------
# Tab Content
# -----------------------------------------------------------

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def render_tab(tab):
    if tab == "ml":
        return dbc.Card(dbc.CardBody([
            html.H4("Machine Learning Scoring"),
            dbc.Input(id="ml-team", placeholder="Team Name", type="text", className="mb-2"),
            dbc.Input(id="ml-accuracy", placeholder="Accuracy", type="number", min=0, max=10, step=0.1, className="mb-2"),
            dbc.Input(id="ml-innovation", placeholder="Innovation", type="number", min=0, max=10, step=0.1, className="mb-2"),
            dbc.Input(id="ml-explainability", placeholder="Explainability", type="number", min=0, max=10, step=0.1, className="mb-2"),
            dbc.Button("Submit ML Score", id="ml-submit", color="primary")
        ]))

    elif tab == "llm":
        return dbc.Card(dbc.CardBody([
            html.H4("LLM Scoring"),
            dbc.Input(id="llm-team", placeholder="Team Name", type="text", className="mb-2"),
            dbc.Input(id="llm-accuracy", placeholder="Accuracy", type="number", min=0, max=10, step=0.1, className="mb-2"),
            dbc.Input(id="llm-innovation", placeholder="Innovation", type="number", min=0, max=10, step=0.1, className="mb-2"),
            dbc.Input(id="llm-explainability", placeholder="Explainability", type="number", min=0, max=10, step=0.1, className="mb-2"),
            dbc.Button("Submit LLM Score", id="llm-submit", color="success")
        ]))

    elif tab == "analysis":
        return dbc.Card(dbc.CardBody([
            html.H4("Analysis Scoring"),
            dbc.Input(id="analysis-team", placeholder="Team Name", type="text", className="mb-2"),
            dbc.Input(id="analysis-depth", placeholder="Depth", type="number", min=0, max=10, step=0.1, className="mb-2"),
            dbc.Input(id="analysis-clarity", placeholder="Clarity", type="number", min=0, max=10, step=0.1, className="mb-2"),
            dbc.Input(id="analysis-creativity", placeholder="Creativity", type="number", min=0, max=10, step=0.1, className="mb-2"),
            dbc.Button("Submit Analysis Score", id="analysis-submit", color="warning")
        ]))

    elif tab == "leaderboard":
        return html.Div([
            html.H4("Leaderboard"),
            html.Div(id="leaderboard"),
            html.Br(),
            dcc.Graph(id="charts")
        ])

    return html.P("Select a tab to continue.")

# -----------------------------------------------------------
# Callbacks for Leaderboard + Charts
# -----------------------------------------------------------

@app.callback(
    [Output("leaderboard", "children"),
     Output("charts", "figure")],
    [Input("ml-submit", "n_clicks"),
     Input("llm-submit", "n_clicks"),
     Input("analysis-submit", "n_clicks")],
    [State("ml-team", "value"),
     State("ml-accuracy", "value"),
     State("ml-innovation", "value"),
     State("ml-explainability", "value"),
     State("llm-team", "value"),
     State("llm-accuracy", "value"),
     State("llm-innovation", "value"),
     State("llm-explainability", "value"),
     State("analysis-team", "value"),
     State("analysis-depth", "value"),
     State("analysis-clarity", "value"),
     State("analysis-creativity", "value")]
)
def update_scores(n1, n2, n3,
                  ml_team, ml_acc, ml_innov, ml_expl,
                  llm_team, llm_acc, llm_innov, llm_expl,
                  an_team, an_depth, an_clarity, an_creativity):

    scores = []

    if ml_team:
        scores.append({
            "Team": ml_team,
            "accuracy": safe_float(ml_acc),
            "innovation": safe_float(ml_innov),
            "explainability": safe_float(ml_expl)
        })

    if llm_team:
        scores.append({
            "Team": llm_team,
            "accuracy": safe_float(llm_acc),
            "innovation": safe_float(llm_innov),
            "explainability": safe_float(llm_expl)
        })

    if an_team:
        scores.append({
            "Team": an_team,
            "depth": safe_float(an_depth),
            "clarity": safe_float(an_clarity),
            "creativity": safe_float(an_creativity)
        })

    if not scores:
        return html.P("No scores yet."), {}

    df = pd.DataFrame(scores)

    # Leaderboard table
    table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)

    # Simple chart: sum of scores per team
    df["total"] = df.drop(columns=["Team"]).sum(axis=1)
    fig = df.plot.bar(x="Team", y="total", legend=False, title="Total Scores").get_figure()

    return table, fig

# -----------------------------------------------------------
# Run
# -----------------------------------------------------------

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)

