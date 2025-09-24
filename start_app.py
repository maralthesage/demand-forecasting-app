#!/usr/bin/env python3
"""
Application startup script for continuous web app
"""
import os
import sys
import subprocess
import argparse
import signal
import threading
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_environment():
    """Set up environment variables and paths"""
    if "SALES_FORECAST_DATA_PATH" not in os.environ:
        os.environ["SALES_FORECAST_DATA_PATH"] = "Z:/"

    os.environ["PYTHONPATH"] = str(project_root)

    from config import create_directories

    create_directories()

    print("✅ Environment setup complete")


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import xgboost
        import lightgbm
        import sklearn
        import schedule
        import psutil

        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False


def start_continuous_app(host="0.0.0.0", port=8501):
    """Start the continuous Streamlit application"""
    print(f"🚀 Starting continuous web app on {host}:{port}")
    print("🤖 Background processing will start automatically")
    print(f"🌐 Access the app at: http://{host}:{port}")

    cmd = [
        "streamlit",
        "run",
        "continuous_app.py",
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main startup function"""
    print("🚀 Demand Forecasting Application")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="Start Demand Forecasting App")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8501, help="Port number")
    parser.add_argument("--data-path", help="Override data path")
    parser.add_argument(
        "--daily-time", default="06:00", help="Daily processing time (HH:MM)"
    )

    args = parser.parse_args()

    # Setup
    setup_environment()

    if args.data_path:
        os.environ["SALES_FORECAST_DATA_PATH"] = args.data_path

    if not check_dependencies():
        return

    # Set daily processing time
    os.environ["DAILY_PROCESSING_TIME"] = args.daily_time

    # Start continuous app
    start_continuous_app(args.host, args.port)


if __name__ == "__main__":
    main()
