# 🚀 Railway Deployment Guide

## Step 1: Create Railway Account

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub (recommended)
3. Verify your email

## Step 2: Deploy from GitHub

### Option A: Deploy from GitHub Repository

1. **Push to GitHub**:
   ```bash
   cd hackathon-deploy
   git init
   git add .
   git commit -m "Initial hackathon webapp deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/hackathon-webapp.git
   git push -u origin main
   ```

2. **Connect to Railway**:
   - Go to Railway dashboard
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will auto-detect Python and deploy

### Option B: Deploy from Local Files

1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and Deploy**:
   ```bash
   cd hackathon-deploy
   railway login
   railway init
   railway up
   ```

## Step 3: Configure Environment Variables

In Railway dashboard:

1. Go to your project
2. Click on "Variables" tab
3. Add environment variables:
   - `DAGSHUB_TOKEN`: Your DagsHub API token (optional)
   - `PORT`: Railway sets this automatically

## Step 4: Get Your Public URL

1. Railway will provide a URL like: `https://your-app-name.railway.app`
2. Your webapp will be live at this URL
3. You can also set up a custom domain in Railway settings

## Step 5: Test Your Deployment

1. **Visit your URL** to see the webapp
2. **Test admin authentication** with password `1998`
3. **Submit test scores** to verify CSV saving works
4. **Check leaderboard** updates in real-time

## 🔧 Troubleshooting

### Common Issues:

1. **Build Fails**:
   - Check `requirements.txt` has all dependencies
   - Ensure `Procfile` is correct
   - Check Railway logs for errors

2. **App Won't Start**:
   - Verify `app.py` has `server = app.server`
   - Check port configuration
   - Review Railway deployment logs

3. **CSV Files Not Saving**:
   - Railway has ephemeral file system
   - Consider using Railway's database or external storage
   - For demo purposes, CSV works but data resets on redeploy

### Railway Logs:
```bash
railway logs
```

## 🎯 Next Steps

1. **Set up DagsHub integration** (optional):
   - Add your DagsHub token to environment variables
   - Update the API calls in `app.py`

2. **Custom domain** (optional):
   - Go to Railway project settings
   - Add custom domain
   - Update DNS records

3. **Database integration** (for production):
   - Consider PostgreSQL for persistent data
   - Railway offers managed databases

## 📊 Monitoring

Railway provides:
- **Deployment logs**
- **Performance metrics**
- **Uptime monitoring**
- **Automatic restarts**

---

**Your hackathon webapp is now live and ready!** 🎉
