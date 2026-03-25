# Demand Forecasting App

This project is a web application for forecasting product demand from historical sales data. It combines data processing, machine learning models, and a simple interface for exploring results.

## Overview

* End-to-end pipeline from raw data to forecasts
* Ensemble models (XGBoost, LightGBM, Random Forest, linear models)
* Streamlit dashboard for visualisation
* Incremental data processing and scheduled updates

## Setup

### Docker

```bash
git clone https://github.com/your-username/demand-forecasting-app.git
cd demand-forecasting-app
docker-compose up -d
```

Open: http://localhost:8501

### Local

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp env.example .env
python start_app.py
```

## Data Format

```csv
product_id,MONAT,anz_produkt,unit_preis
PROD001,2023-01-01,45,12.50
```

Required:

* `product_id`
* `MONAT`
* `anz_produkt`

## Configuration

```bash
SALES_FORECAST_DATA_PATH=/path/to/data
DAILY_PROCESSING_TIME=02:00
```

## Notes

* Requires several months of historical data
* Data quality directly affects forecast accuracy

## License

MIT License
