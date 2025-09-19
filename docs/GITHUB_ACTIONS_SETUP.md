# GitHub Actions CI/CD Setup Guide

This guide will help you set up GitHub Actions for automated deployment and daily data processing of your Sales Forecast application.

## 🔐 Required GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions, and add these secrets:

### Data Access
```
SALES_FORECAST_DATA_PATH=/path/to/your/data/volume
```
- Path to your data directory (where CSV files are located)

### AWS Configuration (if using cloud deployment)
```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_BUCKET=your-sales-forecast-bucket
ECS_CLUSTER_NAME=your-ecs-cluster
ECS_SERVICE_NAME=sales-forecast-service
```

### Notifications (optional)
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Container Registry (if using private registry)
```
DOCKER_REGISTRY=your-registry.com
DOCKER_USERNAME=your-username
DOCKER_PASSWORD=your-password
```

## 🚀 Workflow Features

The GitHub Actions workflow (`deploy.yml`) provides:

### 1. **Continuous Integration**
- Runs on every push to `main` and `develop` branches
- Code linting with `flake8` and `black`
- Automated testing with `pytest`
- Coverage reporting

### 2. **Automated Building**
- Docker image building with caching
- Multi-stage builds for optimization
- Artifact storage for deployment

### 3. **Daily Data Processing**
- Scheduled daily at 1 AM UTC
- Incremental data updates
- Model retraining when needed
- Processing reports and notifications

### 4. **Deployment Automation**
- Staging deployment on `develop` branch
- Production deployment on `main` branch
- Health checks and rollback capability
- Deployment tagging

## 🎯 Trigger Options

### Automatic Triggers
- **Push to main/develop**: Triggers build and deployment
- **Pull Request**: Runs tests and creates preview
- **Daily Schedule**: Runs data processing at 1 AM UTC
- **Commit with `[data-update]`**: Forces data processing

### Manual Triggers
- **Workflow Dispatch**: Manual trigger with options
  - Force model retraining
  - Choose deployment environment

## 📋 Workflow Jobs

### 1. **Test Job**
```yaml
- Checkout code
- Set up Python 3.9
- Install dependencies
- Run linting (flake8, black)
- Run tests with coverage
- Upload coverage reports
```

### 2. **Build Job**
```yaml
- Build Docker image
- Cache layers for faster builds
- Store image as artifact
- Extract metadata and tags
```

### 3. **Data Processing Job**
```yaml
- Run on schedule or manual trigger
- Process new data incrementally
- Train models for new products
- Create processing reports
- Archive results
```

### 4. **Deploy Staging**
```yaml
- Deploy to staging environment
- Run health checks
- Notify on status
```

### 5. **Deploy Production**
```yaml
- Deploy to production environment
- Create deployment tags
- Run comprehensive health checks
- Notify stakeholders
```

## 🔧 Environment Setup

### Local Development
```bash
# Copy environment template
cp env.example .env

# Edit with your values
nano .env

# Install dependencies
pip install -r requirements.txt

# Run locally
python start_app.py
```

### Production Server
```bash
# Clone repository
git clone https://github.com/your-username/sales_forecast_app.git
cd sales_forecast_app

# Set up environment
cp env.example .env
nano .env  # Edit with production values

# Start with Docker Compose
docker-compose up -d
```

## 📊 Monitoring Workflow

### Check Workflow Status
1. Go to your repository on GitHub
2. Click "Actions" tab
3. View workflow runs and their status

### View Processing Reports
- Processing results are stored as artifacts
- Download artifacts to view detailed reports
- Check logs for troubleshooting

### Notifications
- Configure Slack webhook for real-time notifications
- Receive alerts on deployment success/failure
- Get daily processing reports

## 🛠️ Customization

### Modify Schedule
Edit `.github/workflows/deploy.yml`:
```yaml
schedule:
  # Change to your preferred time (UTC)
  - cron: '0 2 * * *'  # 2 AM UTC instead of 1 AM
```

### Add Custom Steps
```yaml
- name: Custom Step
  run: |
    echo "Your custom commands here"
    python your_script.py
```

### Environment-Specific Configuration
```yaml
environment: production  # or staging
```

## 🚨 Troubleshooting

### Common Issues

1. **Secrets Not Found**
   - Verify secrets are added to repository settings
   - Check secret names match exactly

2. **Data Access Errors**
   - Ensure `SALES_FORECAST_DATA_PATH` is correct
   - Check file permissions

3. **Build Failures**
   - Review build logs in Actions tab
   - Check Docker configuration

4. **Deployment Issues**
   - Verify cloud credentials
   - Check service configurations

### Debug Commands
```bash
# Test locally
python -m pytest tests/ -v

# Check Docker build
docker build -t test-app .

# Validate environment
python -c "import os; print(os.getenv('SALES_FORECAST_DATA_PATH'))"
```

## 📈 Performance Optimization

### Caching Strategy
- Docker layer caching
- Python dependency caching
- Model and data caching

### Resource Management
- Adjust memory limits in workflow
- Optimize Docker image size
- Use multi-stage builds

## 🔒 Security Best Practices

1. **Never commit secrets** to repository
2. **Use environment-specific secrets**
3. **Regularly rotate access keys**
4. **Limit workflow permissions**
5. **Review workflow logs** for sensitive data

## 📞 Support

If you encounter issues:
1. Check the [troubleshooting section](#🚨-troubleshooting)
2. Review workflow logs in GitHub Actions
3. Check repository issues for similar problems
4. Create a new issue with detailed error information
