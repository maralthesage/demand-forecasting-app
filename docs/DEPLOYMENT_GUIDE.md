# Sales Forecast App - Deployment Guide

This guide covers deployment options for your continuous Sales Forecast application.

## 🚀 Quick Start

### Option 1: Local Deployment with Docker
```bash
# Clone repository
git clone https://github.com/your-username/sales_forecast_app.git
cd sales_forecast_app

# Set up environment
cp env.example .env
nano .env  # Configure your settings

# Start application
docker-compose up -d

# Access app at http://localhost:8501
```

### Option 2: Server Deployment
```bash
# On your server
git clone https://github.com/your-username/sales_forecast_app.git
cd sales_forecast_app

# Configure environment
cp env.example .env
nano .env

# Start application
python start_app.py --host 0.0.0.0 --port 8501
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │    │  GitHub Actions │    │   Application   │
│   (CSV Files)   │───▶│   (CI/CD)       │───▶│   (Streamlit)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Notifications  │    │  Background     │    │     Cache       │
│    (Slack)      │◀───│  Scheduler      │◀───│  (Models/Data)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with your configuration:

```bash
# Data Configuration
SALES_FORECAST_DATA_PATH=/path/to/your/data
DAILY_PROCESSING_TIME=02:00

# Application Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
LOG_LEVEL=INFO

# Optional: Cloud Storage
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET=your-bucket

# Optional: Notifications
SLACK_WEBHOOK_URL=your_webhook_url
```

### Data Structure
Ensure your data directory has this structure:
```
/path/to/your/data/
├── CSV/
│   ├── F01/
│   │   ├── V2SC1010.csv  # Nachfrage data
│   │   ├── V2AR1001.csv  # Marketing artikel
│   │   └── V2AR1002.csv  # Product descriptions
│   ├── F02/
│   │   └── V2SC1010.csv
│   ├── F03/
│   │   └── V2SC1010.csv
│   └── F04/
│       └── V2SC1010.csv
└── Data/
    └── sales_forecast/  # Processed data output
```

## 🐳 Docker Deployment

### Single Container
```bash
# Build image
docker build -t sales-forecast-app .

# Run container
docker run -d \
  --name sales-forecast \
  -p 8501:8501 \
  -v /path/to/data:/data \
  -v $(pwd)/cache:/app/cache \
  -e SALES_FORECAST_DATA_PATH=/data \
  -e DAILY_PROCESSING_TIME=02:00 \
  sales-forecast-app
```

### Docker Compose (Recommended)
```yaml
# docker-compose.yml
version: '3.8'
services:
  sales-forecast-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - SALES_FORECAST_DATA_PATH=/data
      - DAILY_PROCESSING_TIME=02:00
    volumes:
      - ./cache:/app/cache
      - ./logs:/app/logs
      - /path/to/your/data:/data
    restart: unless-stopped
```

Start with:
```bash
docker-compose up -d
```

## ☁️ Cloud Deployment Options

### AWS ECS/Fargate

1. **Create ECR Repository**
```bash
aws ecr create-repository --repository-name sales-forecast-app
```

2. **Push Image**
```bash
# Build and tag
docker build -t sales-forecast-app .
docker tag sales-forecast-app:latest $ECR_URI:latest

# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
docker push $ECR_URI:latest
```

3. **Create ECS Service**
- Use the provided GitHub Actions workflow
- Configure task definition with your image
- Set up load balancer and auto-scaling

### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/sales-forecast-app
gcloud run deploy --image gcr.io/PROJECT-ID/sales-forecast-app --platform managed
```

### Azure Container Instances

```bash
# Create resource group
az group create --name sales-forecast-rg --location eastus

# Deploy container
az container create \
  --resource-group sales-forecast-rg \
  --name sales-forecast-app \
  --image your-registry/sales-forecast-app:latest \
  --ports 8501 \
  --environment-variables SALES_FORECAST_DATA_PATH=/data
```

## 🔄 Continuous Deployment

### GitHub Actions Setup

1. **Configure Repository Secrets**
   - See [GitHub Actions Setup Guide](GITHUB_ACTIONS_SETUP.md)

2. **Workflow Triggers**
   - Push to `main`: Production deployment
   - Push to `develop`: Staging deployment
   - Daily schedule: Data processing
   - Manual dispatch: On-demand operations

3. **Deployment Process**
   ```
   Code Push → Build → Test → Deploy → Health Check → Notify
   ```

## 📊 Monitoring & Maintenance

### Health Checks
The application provides health endpoints:
- `http://your-app:8501/_stcore/health` - Streamlit health
- Application includes built-in system status monitoring

### Logs
```bash
# View application logs
docker logs sales-forecast-app

# View processing logs
tail -f logs/app.log

# View system status
curl http://localhost:8501/system-status
```

### Backup Strategy
```bash
# Backup cache and models
tar -czf backup-$(date +%Y%m%d).tar.gz cache/ models/

# Restore from backup
tar -xzf backup-20241201.tar.gz
```

## 🔧 Troubleshooting

### Common Issues

1. **App Won't Start**
   ```bash
   # Check logs
   docker logs sales-forecast-app
   
   # Verify environment
   docker exec -it sales-forecast-app env | grep SALES
   ```

2. **Data Not Loading**
   ```bash
   # Check data path
   docker exec -it sales-forecast-app ls -la /data
   
   # Test data access
   python -c "from data.data_loader import DataLoader; DataLoader().load_nachfrage_data()"
   ```

3. **Processing Failures**
   ```bash
   # Force manual processing
   docker exec -it sales-forecast-app python -c "from background_scheduler import background_scheduler; background_scheduler.force_run_now()"
   ```

4. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   docker update --memory=4g sales-forecast-app
   
   # Or in docker-compose.yml:
   mem_limit: 4g
   ```

### Performance Tuning

1. **Optimize Data Loading**
   - Use parquet format for faster loading
   - Implement data chunking for large datasets
   - Cache frequently accessed data

2. **Model Training**
   - Adjust `min_history_months` parameter
   - Use incremental training for new products only
   - Implement model versioning

3. **Resource Management**
   - Monitor memory usage during processing
   - Adjust processing schedule based on data size
   - Use SSD storage for cache directory

## 🚨 Production Checklist

### Pre-Deployment
- [ ] Environment variables configured
- [ ] Data paths accessible
- [ ] Backup strategy in place
- [ ] Monitoring configured
- [ ] SSL certificates (if needed)
- [ ] Firewall rules configured

### Post-Deployment
- [ ] Health checks passing
- [ ] Data processing working
- [ ] Notifications configured
- [ ] Performance monitoring active
- [ ] Backup schedule verified
- [ ] Documentation updated

## 🔐 Security Considerations

1. **Data Protection**
   - Encrypt data at rest
   - Use secure data transfer
   - Implement access controls

2. **Application Security**
   - Keep dependencies updated
   - Use non-root user in containers
   - Implement rate limiting

3. **Network Security**
   - Use HTTPS in production
   - Configure firewall rules
   - Monitor access logs

## 📈 Scaling

### Horizontal Scaling
- Deploy multiple instances behind load balancer
- Use shared storage for cache and models
- Implement session affinity if needed

### Vertical Scaling
- Increase memory for larger datasets
- Add CPU cores for faster processing
- Use SSD storage for better I/O

## 📞 Support

For deployment issues:
1. Check logs first: `docker logs sales-forecast-app`
2. Verify configuration: Review `.env` file
3. Test data access: Ensure data paths are correct
4. Review GitHub Actions logs for CI/CD issues
5. Create GitHub issue with detailed error information
