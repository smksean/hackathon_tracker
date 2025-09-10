# 🏆 UK Housing Market Hackathon Webapp

A real-time leaderboard and scoring interface for the UK Housing Market hackathon, deployed on Railway.

## 🚀 Live Demo

**Access the live webapp**: [Your Railway URL will appear here after deployment]

## 🎯 Features

### 📊 Live Leaderboard
- **Real-time updates** every 30 seconds
- **Smart aggregate scoring** combining ML, LLM, and Analysis scores
- **Team rankings** with visual indicators (🥇🥈🥉)
- **Performance metrics** display for each team

### 🔐 Admin Scoring Interface
- **Password protection** (password: 1998)
- **Three scoring categories**:
  - **ML Model Scoring**: R², RMSE, MSE from DagsHub
  - **LLM Implementation**: Creativity, Accuracy, Inference Speed
  - **Analysis Quality**: Quality, Completeness, Innovation
- **CSV persistence** - all scores automatically saved (with sample data included)

### 📈 Performance Analytics
- **Total Score distribution** chart
- **LLM Speed vs Accuracy** trade-off visualization
- **Interactive charts** with hover details

## 🛠️ Technical Stack

- **Frontend**: Dash + Bootstrap
- **Backend**: Python + Flask
- **Deployment**: Railway
- **Data Storage**: CSV files (with sample data for demo)
- **Real-time Updates**: Dash intervals

## 🔧 Environment Variables

Set these in your Railway dashboard:

- `DAGSHUB_TOKEN`: Your DagsHub API token (optional)
- `PORT`: Railway sets this automatically

## 📱 Usage

### For Hackathon Participants
1. **View Leaderboard**: See real-time team rankings
2. **Check Performance**: View your team's scores and metrics

### For Hackathon Hosts
1. **Authenticate**: Enter password `1998`
2. **Score Teams**: Use the three scoring interfaces
3. **Monitor Progress**: Watch the live leaderboard update

## 🚀 Deployment

This app is deployed on Railway with:
- **Automatic deployments** from GitHub
- **Environment variable** support
- **Persistent file storage** for scores
- **Custom domain** support

## 📊 Scoring System

### Composite Score Calculation
```
Total Score = (ML Score × 50%) + (LLM Score × 30%) + (Analysis Score × 20%)
```

### Individual Scores
- **ML Score (0-100)**: Based on R², RMSE, MSE
- **LLM Score (0-100)**: Creativity, Accuracy, Speed
- **Analysis Score (0-100)**: Quality, Completeness, Innovation

## 🎨 UI Features

- **Professional design** with consistent colors
- **Responsive layout** for all devices
- **Real-time updates** with live clock
- **Admin authentication** for secure scoring
- **Interactive charts** and visualizations

---

**Built for the UK Housing Market Hackathon** 🏠
