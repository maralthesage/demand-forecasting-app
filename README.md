# 🚀 Open Source Demand Forecasting App

> **Predict your product demand with AI-powered forecasting**

An intelligent web application that helps businesses forecast product demand using advanced machine learning. Whether you're managing inventory, planning production, or making strategic decisions, this app provides accurate predictions with confidence intervals.

[![Deploy](https://github.com/your-username/demand-forecasting-app/actions/workflows/deploy.yml/badge.svg)](https://github.com/your-username/demand-forecasting-app/actions/workflows/deploy.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What This App Does

**In simple terms**: Upload your sales data → Get accurate demand forecasts → Make better business decisions.

- **📊 Smart Predictions**: Uses ensemble machine learning (XGBoost, LightGBM, Random Forest) for accurate forecasts
- **🔄 Always Up-to-Date**: Automatically processes new data daily at 2 AM
- **📈 Interactive Dashboard**: Beautiful charts and insights you can explore
- **⚡ Fast & Efficient**: Only processes new data, not everything from scratch
- **🎛️ Easy to Use**: Web interface - no coding required for daily use
- **🔒 Secure**: Your data stays private and secure

## 🚀 Quick Start (5 Minutes)

### Try the Demo First
```bash
# 1. Clone the repository
git clone https://github.com/your-username/demand-forecasting-app.git
cd demand-forecasting-app

# 2. Run with Docker (easiest way)
docker-compose up -d

# 3. Open your browser
# Go to: http://localhost:8501
```

That's it! The app will start with sample data so you can see how it works.

### Use Your Own Data
1. **Prepare your data**: CSV files with columns: `product_id`, `date`, `sales_quantity`, `unit_price`
2. **Set data path**: Copy `env.example` to `.env` and set your data path
3. **Restart the app**: It will automatically detect and process your data

## 📊 What You'll See

### Live Dashboard
- **Real-time metrics**: Total sales, top products, trends
- **Interactive charts**: Click and explore your data
- **Data freshness**: Always know how current your data is

### Demand Forecasting
- **Select any product**: Get instant forecasts
- **Flexible horizons**: Predict 1-12 months ahead
- **Confidence bands**: Know the uncertainty in predictions
- **Downloadable results**: Export forecasts to CSV

### System Monitoring
- **Processing status**: See when data was last updated
- **Manual controls**: Force updates or refresh cache
- **Health checks**: Monitor system performance

## 🛠️ Installation Options

### Option 1: Docker (Recommended)
Perfect for trying the app or production deployment.

```bash
# Clone and start
git clone https://github.com/your-username/demand-forecasting-app.git
cd demand-forecasting-app
docker-compose up -d

# Access at http://localhost:8501
```

### Option 2: Python Environment
For developers or custom setups.

```bash
# Clone repository
git clone https://github.com/your-username/demand-forecasting-app.git
cd demand-forecasting-app

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with your settings

# Start application
python start_app.py
```

### Option 3: Cloud Deployment
Deploy to AWS, Google Cloud, or Azure using the included GitHub Actions workflow.

## 📁 Your Data Format

The app works with standard sales data. Here's what it expects:

### CSV Structure
```csv
product_id,product_category_id,MONAT,anz_produkt,unit_preis
PROD001,CATEGORY1,2023-01-01,45,12.50
PROD001,CATEGORY1,2023-02-01,52,12.50
PROD002,CATEGORY2,2023-01-01,23,8.90
```

### Required Columns
- **product_id**: Unique identifier for each product
- **MONAT**: Date (YYYY-MM-DD format, typically first day of month)
- **anz_produkt**: Sales quantity/demand
- **unit_preis**: Unit price

### Optional Columns
- **product_category_id**: Product category (helps with new product forecasts)
- **region**: Geographic region (for multi-region analysis)

### File Organization
```
your_data_folder/
├── CSV/
│   ├── F01/V2SC1010.csv  # Region 1 sales data
│   ├── F02/V2SC1010.csv  # Region 2 sales data
│   └── ...
└── processed/            # App will create this
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file with your settings:

```bash
# Required: Path to your data
SALES_FORECAST_DATA_PATH=/path/to/your/data

# Optional: When to run daily processing (24-hour format)
DAILY_PROCESSING_TIME=02:00

# Optional: Notifications
SLACK_WEBHOOK_URL=your_slack_webhook_url
```

### Advanced Settings
Edit `config.py` for advanced configuration:
- Forecast horizon (default: 6 months)
- Minimum history required (default: 6 months)
- Model parameters
- Feature engineering settings

## 🤖 How the AI Works

### Machine Learning Models
The app uses an ensemble of proven forecasting models:
- **XGBoost**: Excellent for complex patterns
- **LightGBM**: Fast and accurate
- **Random Forest**: Robust to outliers
- **Linear Models**: Baseline and interpretability

### Smart Features
The AI automatically creates features from your data:
- **Seasonal patterns**: Detects monthly/quarterly cycles
- **Trends**: Identifies growth or decline patterns
- **Price effects**: Considers price changes on demand
- **Lag features**: Uses historical values for prediction
- **Category insights**: Learns from similar products

### Automatic Processing
Every day at 2 AM (configurable):
1. **Scans for new data** in your CSV files
2. **Processes only new records** (super efficient!)
3. **Updates forecasts** for all products
4. **Refreshes the dashboard** with latest insights

## 📈 Business Use Cases

### Inventory Management
- **Stock optimization**: Know how much to order
- **Reduce waste**: Avoid overstock situations
- **Prevent stockouts**: Ensure popular items are available

### Production Planning
- **Capacity planning**: Schedule production based on demand
- **Resource allocation**: Assign staff and materials efficiently
- **Seasonal preparation**: Get ready for peak seasons

### Strategic Planning
- **Budget forecasting**: Predict revenue from demand forecasts
- **New product insights**: Understand category performance
- **Market analysis**: Identify trends and opportunities

## 🔧 Troubleshooting

### Common Issues

**App won't start?**
```bash
# Check if port 8501 is already in use
lsof -i :8501

# Or try a different port
python start_app.py --port 8502
```

**No data showing?**
1. Check your data path in `.env` file
2. Ensure CSV files have the right column names
3. Look at the sample data format in `data/sample/`

**Forecasts seem wrong?**
- Ensure you have at least 6 months of historical data
- Check for data quality issues (missing dates, outliers)
- Consider adjusting the minimum history in `config.py`

### Getting Help
- **Check the logs**: Look in `logs/app.log` for error messages
- **System status**: Use the "System Status" tab in the app
- **GitHub Issues**: Report bugs or ask questions
- **Documentation**: See the `docs/` folder for detailed guides

## 🚀 Deployment to Production

### Automated Deployment
The app includes GitHub Actions for automatic deployment:
1. **Push to GitHub**: Triggers automated testing
2. **Deploy to staging**: Test your changes safely
3. **Deploy to production**: With one click approval

### Manual Deployment
For custom server deployment:
```bash
# On your server
git clone https://github.com/your-username/demand-forecasting-app.git
cd demand-forecasting-app

# Configure for production
cp env.example .env
nano .env  # Set your production settings

# Start with Docker
docker-compose -f docker-compose.yml up -d

# Or with systemd service
sudo systemctl enable demand-forecasting-app
sudo systemctl start demand-forecasting-app
```

## 🤝 Contributing

We welcome contributions! This project is open source and community-driven.

### Ways to Contribute
- **🐛 Report bugs**: Found something broken? Let us know!
- **💡 Suggest features**: Have ideas for improvements?
- **📝 Improve docs**: Help make the documentation better
- **🔧 Submit code**: Fix bugs or add new features

### Development Setup
```bash
# Fork the repository on GitHub
git clone https://github.com/your-username/demand-forecasting-app.git
cd demand-forecasting-app

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/

# Format code
black .
flake8 .

# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes and submit a pull request!
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**What this means**: You can use, modify, and distribute this software freely, even for commercial purposes. Just keep the license notice.

## 🙏 Acknowledgments

Built with love using these amazing open source tools:
- **[Streamlit](https://streamlit.io/)** - For the beautiful web interface
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning foundation
- **[XGBoost](https://xgboost.readthedocs.io/)** & **[LightGBM](https://lightgbm.readthedocs.io/)** - Powerful ML models
- **[Plotly](https://plotly.com/)** - Interactive charts and visualizations
- **[Docker](https://www.docker.com/)** - Easy deployment and scaling

## 📞 Support & Community

- **🐛 Issues**: [GitHub Issues](https://github.com/your-username/demand-forecasting-app/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/your-username/demand-forecasting-app/discussions)
- **📚 Wiki**: [Project Wiki](https://github.com/your-username/demand-forecasting-app/wiki)
- **📧 Email**: your-email@domain.com

---

**Ready to forecast your demand?** 
[⭐ Star this repo](https://github.com/your-username/demand-forecasting-app) • [🚀 Try the demo](http://localhost:8501) • [📖 Read the docs](docs/)

*Made with ❤️ for businesses who want to make data-driven decisions*