#!/bin/bash

echo "🏭 Sales Forecast Application Setup"
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs models cache data output

# Set permissions
echo "🔐 Setting permissions..."
chmod +x start_app.py
chmod +x run_pipeline.py

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 To start the application:"
echo "   source .venv/bin/activate"
echo "   python start_app.py"
echo ""
echo "🔧 To run the full pipeline:"
echo "   source .venv/bin/activate" 
echo "   python run_pipeline.py --mode all"
echo ""
echo "🐳 To run with Docker:"
echo "   docker-compose up --build"
echo ""
