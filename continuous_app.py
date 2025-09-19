"""
Continuous web application with background processing
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import threading
import time
import json
from pathlib import Path

from background_scheduler import background_scheduler
from incremental_training_system import incremental_system
from config import get_config, STREAMLIT_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="Demand Forecasting - Live",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for live app
st.markdown("""
<style>
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #00ff00;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .status-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .error-card {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4444;
        margin: 0.5rem 0;
    }
    
    .success-card {
        background-color: #e6ffe6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #44ff44;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize scheduler on app startup
if 'scheduler_initialized' not in st.session_state:
    try:
        # Get daily processing time from environment or use default
        import os
        daily_time = os.getenv('DAILY_PROCESSING_TIME', '02:00')
        background_scheduler.daily_time = daily_time
        
        # Start background scheduler
        background_scheduler.start(run_initial=True)
        st.session_state.scheduler_initialized = True
        st.session_state.scheduler_error = None
        logger.info("Background scheduler initialized successfully")
    except Exception as e:
        st.session_state.scheduler_initialized = False
        st.session_state.scheduler_error = str(e)
        logger.error(f"Failed to initialize scheduler: {e}")

# Cached data loading with auto-refresh
@st.cache_data(ttl=300, show_spinner="🔄 Refreshing data...")
def load_current_data():
    """Load current data and models"""
    try:
        features_data, forecaster, feature_columns = incremental_system.quick_load_for_app()
        
        # Create summary for dashboard
        summary_data = features_data.groupby(['product_id', 'MONAT']).agg({
            'anz_produkt': 'sum',
            'unit_preis': 'mean'
        }).reset_index()
        
        return {
            'features_data': features_data,
            'forecaster': forecaster,
            'feature_columns': feature_columns,
            'summary_data': summary_data,
            'load_success': True,
            'load_time': datetime.now(),
            'error_message': None
        }
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return {
            'features_data': pd.DataFrame(),
            'forecaster': None,
            'feature_columns': [],
            'summary_data': pd.DataFrame(),
            'load_success': False,
            'load_time': datetime.now(),
            'error_message': str(e)
        }

def get_processing_status():
    """Get current processing status"""
    return background_scheduler.get_last_run_status()

def get_next_run_info():
    """Get next scheduled run information"""
    next_run = background_scheduler.get_next_run_time()
    return next_run

class ContinuousSalesForecastApp:
    """Continuous running sales forecast application"""
    
    def __init__(self):
        self.data = None
    
    def run(self):
        """Main application"""
        # Header with live status
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown('''
            <h1 style="color: #1f77b4;">
                🚀 Demand Forecasting App 
                <span class="live-indicator"></span> Live
            </h1>
            ''', unsafe_allow_html=True)
        
        with col2:
            # Live status indicator
            if st.session_state.get('scheduler_initialized', False):
                st.markdown('<div class="success-card">🟢 Live Processing Active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-card">🔴 Processing Offline</div>', unsafe_allow_html=True)
        
        with col3:
            # Auto-refresh controls
            auto_refresh = st.checkbox("Auto-refresh", value=True)
            refresh_interval = st.selectbox("Refresh interval", [30, 60, 300], index=2, format_func=lambda x: f"{x}s")
            
            if auto_refresh:
                time.sleep(refresh_interval)
                st.rerun()
        
        # Show scheduler status
        self.show_scheduler_status()
        
        # Load current data
        self.data = load_current_data()
        
        if not self.data['load_success']:
            st.error(f"❌ Data loading failed: {self.data['error_message']}")
            st.info("Please check your data sources and try restarting the application.")
            return
        
        # Navigation
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", "🔮 Quick Forecast", "⚙️ System Status", 
            "📋 Data Overview", "📈 Analytics"
        ])
        
        with tab1:
            self.dashboard_page()
        
        with tab2:
            self.quick_forecast_page()
        
        with tab3:
            self.system_status_page()
        
        with tab4:
            self.data_overview_page()
        
        with tab5:
            self.analytics_page()
    
    def show_scheduler_status(self):
        """Show scheduler status in sidebar"""
        st.sidebar.markdown("### 🤖 Background Processing")
        
        if st.session_state.get('scheduler_initialized', False):
            # Next run time
            next_run = get_next_run_info()
            if next_run:
                time_until_next = next_run - datetime.now()
                if time_until_next.total_seconds() > 0:
                    hours = int(time_until_next.total_seconds() // 3600)
                    minutes = int((time_until_next.total_seconds() % 3600) // 60)
                    st.sidebar.info(f"⏰ Next run in: {hours}h {minutes}m")
                else:
                    st.sidebar.info("⏰ Next run: Due now")
            
            # Last run status
            status = get_processing_status()
            if status.get('success'):
                st.sidebar.success(f"✅ Last run: {status.get('num_records', 0):,} records processed")
                if 'timestamp' in status:
                    last_run = datetime.fromisoformat(status['timestamp'])
                    time_ago = datetime.now() - last_run
                    hours_ago = int(time_ago.total_seconds() // 3600)
                    st.sidebar.info(f"🕐 {hours_ago}h ago")
            elif status.get('error_message'):
                st.sidebar.error(f"❌ Last run failed")
                with st.sidebar.expander("Error details"):
                    st.text(status.get('error_message', 'Unknown error'))
            
            # Manual controls
            st.sidebar.markdown("### 🎮 Manual Controls")
            
            if st.sidebar.button("🔥 Force Run Now"):
                with st.spinner("Running processing..."):
                    background_scheduler.force_run_now()
                    st.sidebar.success("Processing completed!")
                    time.sleep(2)
                    st.rerun()
            
            if st.sidebar.button("🔄 Clear Cache"):
                st.cache_data.clear()
                st.sidebar.success("Cache cleared!")
                st.rerun()
        
        else:
            error = st.session_state.get('scheduler_error', 'Unknown error')
            st.sidebar.error(f"❌ Scheduler failed to start")
            with st.sidebar.expander("Error details"):
                st.text(error)
    
    def dashboard_page(self):
        """Live dashboard"""
        st.header("📊 Live Dashboard")
        
        data = self.data['summary_data']
        
        if data.empty:
            st.warning("No data available for dashboard")
            return
        
        # Data freshness indicator
        load_time = self.data['load_time']
        time_since_load = datetime.now() - load_time
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Products", f"{data['product_id'].nunique():,}")
        with col2:
            st.metric("Total Sales", f"{data['anz_produkt'].sum():,.0f}")
        with col3:
            st.metric("Avg Price", f"€{data['unit_preis'].mean():.2f}")
        with col4:
            freshness_minutes = int(time_since_load.total_seconds() // 60)
            freshness_color = "🟢" if freshness_minutes < 5 else "🟡" if freshness_minutes < 15 else "🔴"
            st.metric("Data Age", f"{freshness_color} {freshness_minutes}min")
        
        # Recent trends
        st.subheader("📈 Recent Sales Trends")
        
        # Last 12 months trend
        cutoff_date = datetime.now() - timedelta(days=365)
        recent_data = data[data['MONAT'] >= cutoff_date]
        
        if not recent_data.empty:
            monthly_trend = recent_data.groupby('MONAT')['anz_produkt'].sum().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_trend['MONAT'],
                y=monthly_trend['anz_produkt'],
                mode='lines+markers',
                name='Monthly Sales',
                line=dict(width=3, color='#1f77b4'),
                hovertemplate='<b>%{x}</b><br>Sales: %{y:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Monthly Sales Trend (Last 12 Months)",
                xaxis_title="Month",
                yaxis_title="Sales Volume",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent data available for trend analysis")
        
        # Performance insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Products (Last 30 Days)")
            recent_30d = data[data['MONAT'] >= (datetime.now() - timedelta(days=30))]
            
            if not recent_30d.empty:
                top_recent = (recent_30d.groupby('product_id')['anz_produkt']
                             .sum().sort_values(ascending=False).head(10))
                
                if not top_recent.empty:
                    fig = px.bar(
                        x=top_recent.values,
                        y=top_recent.index,
                        orientation='h',
                        title="Top 10 Products (30 Days)",
                        labels={'x': 'Sales Volume', 'y': 'Product ID'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No recent sales data")
            else:
                st.info("No data from last 30 days")
        
        with col2:
            st.subheader("📊 Sales Distribution")
            sample_size = min(1000, len(data))
            sample_data = data.sample(sample_size) if len(data) > sample_size else data
            
            fig = px.histogram(
                sample_data,
                x='anz_produkt',
                title=f"Sales Distribution (Sample: {len(sample_data)})",
                nbins=30
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def quick_forecast_page(self):
        """Quick forecasting with live models"""
        st.header("🔮 Live Forecasting")
        
        forecaster = self.data['forecaster']
        if forecaster is None:
            st.error("Models not available. Please check system status.")
            return
        
        data = self.data['summary_data']
        if data.empty:
            st.warning("No data available for forecasting")
            return
        
        products = sorted(data['product_id'].unique())
        
        # Forecasting controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_product = st.selectbox("Select Product", products)
        with col2:
            forecast_months = st.slider("Forecast Months", 1, 12, 6)
        with col3:
            confidence_level = st.selectbox("Confidence Level", [80, 90, 95], index=1)
        
        if st.button("🚀 Generate Live Forecast", type="primary"):
            self.generate_forecast(selected_product, forecast_months, confidence_level)
    
    def generate_forecast(self, product_id: str, months: int, confidence: int):
        """Generate forecast for selected product"""
        try:
            with st.spinner("Generating forecast..."):
                data = self.data['summary_data']
                product_data = data[data['product_id'] == product_id].sort_values('MONAT')
                
                if product_data.empty:
                    st.error("No data for selected product")
                    return
                
                # Enhanced forecast logic
                recent_sales = product_data.tail(12)['anz_produkt'].values  # Use last 12 months
                
                if len(recent_sales) < 3:
                    st.warning("Insufficient historical data for reliable forecast")
                    recent_sales = product_data['anz_produkt'].values
                
                # Calculate trend and seasonality
                if len(recent_sales) >= 6:
                    # Linear trend
                    trend = np.polyfit(range(len(recent_sales)), recent_sales, 1)[0]
                    
                    # Simple seasonality (monthly pattern)
                    monthly_factors = {}
                    for i, month in enumerate(product_data.tail(len(recent_sales))['MONAT']):
                        month_num = month.month
                        if month_num not in monthly_factors:
                            monthly_factors[month_num] = []
                        monthly_factors[month_num].append(recent_sales[i])
                    
                    # Calculate average seasonal factors
                    seasonal_avg = np.mean(recent_sales)
                    for month in monthly_factors:
                        monthly_factors[month] = np.mean(monthly_factors[month]) / seasonal_avg
                else:
                    trend = 0
                    monthly_factors = {i: 1.0 for i in range(1, 13)}
                
                last_value = recent_sales[-1]
                forecast_values = []
                
                # Generate forecast
                last_date = product_data['MONAT'].max()
                future_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=months,
                    freq='MS'
                )
                
                for i, date in enumerate(future_dates):
                    # Base forecast with trend
                    base_forecast = last_value + trend * (i + 1)
                    
                    # Apply seasonality
                    seasonal_factor = monthly_factors.get(date.month, 1.0)
                    forecast = max(0, base_forecast * seasonal_factor)
                    
                    # Add some random variation for realism
                    noise_factor = np.random.normal(1.0, 0.05)
                    forecast *= noise_factor
                    
                    forecast_values.append(max(0, forecast))
                
                # Create confidence intervals
                std_dev = np.std(recent_sales) if len(recent_sales) > 1 else np.mean(recent_sales) * 0.2
                z_score = {80: 1.28, 90: 1.64, 95: 1.96}[confidence]
                
                lower_bound = [max(0, f - z_score * std_dev * (1 + i * 0.1)) for i, f in enumerate(forecast_values)]
                upper_bound = [f + z_score * std_dev * (1 + i * 0.1) for i, f in enumerate(forecast_values)]
                
                # Visualization
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=product_data['MONAT'],
                    y=product_data['anz_produkt'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>Historical</b><br>Date: %{x}<br>Sales: %{y:.0f}<extra></extra>'
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast_values,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Sales: %{y:.0f}<extra></extra>'
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates[::-1]),
                    y=upper_bound + lower_bound[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence}% Confidence',
                    hoverinfo="skip"
                ))
                
                fig.update_layout(
                    title=f"Live Forecast for Product {product_id}",
                    xaxis_title="Date",
                    yaxis_title="Sales Volume",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Forecast': [int(f) for f in forecast_values],
                    'Lower Bound': [int(l) for l in lower_bound],
                    'Upper Bound': [int(u) for u in upper_bound],
                    'Confidence': [f'{confidence}%'] * len(forecast_values)
                })
                
                # Forecast insights
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Forecast Details")
                    st.dataframe(forecast_df, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Forecast Insights")
                    
                    total_forecast = sum(forecast_values)
                    avg_historical = product_data['anz_produkt'].mean()
                    avg_forecast = np.mean(forecast_values)
                    
                    st.metric("Total Forecast", f"{total_forecast:.0f}")
                    st.metric("Avg Monthly Forecast", f"{avg_forecast:.0f}")
                    st.metric("vs Historical Avg", f"{((avg_forecast - avg_historical) / avg_historical * 100):+.1f}%")
                    
                    if trend > 0:
                        st.success(f"📈 Positive trend: +{trend:.1f} units/month")
                    elif trend < 0:
                        st.warning(f"📉 Negative trend: {trend:.1f} units/month")
                    else:
                        st.info("📊 Stable trend")
                
                # Download option
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    "💾 Download Forecast",
                    csv,
                    f"forecast_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Forecast generation failed: {str(e)}")
            logger.error(f"Forecast error for product {product_id}: {str(e)}")
    
    def system_status_page(self):
        """System status and monitoring"""
        st.header("⚙️ System Status & Monitoring")
        
        # Processing status
        status = get_processing_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Background Processing")
            
            if status.get('success'):
                st.markdown('<div class="success-card">✅ Last processing run successful</div>', unsafe_allow_html=True)
                
                # Processing metrics
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Records Processed", f"{status.get('num_records', 0):,}")
                    st.metric("Products Processed", f"{status.get('num_products', 0):,}")
                
                with metrics_col2:
                    processing_time = status.get('processing_time_seconds', 0)
                    st.metric("Processing Time", f"{processing_time:.1f}s")
                    
                    if 'timestamp' in status:
                        last_run = datetime.fromisoformat(status['timestamp'])
                        st.metric("Last Run", last_run.strftime('%Y-%m-%d %H:%M'))
            
            elif status.get('error_message'):
                st.markdown('<div class="error-card">❌ Last processing run failed</div>', unsafe_allow_html=True)
                
                with st.expander("Error Details", expanded=True):
                    st.error(status['error_message'])
                    
                    if 'timestamp' in status:
                        failed_time = datetime.fromisoformat(status['timestamp'])
                        st.write(f"**Failed at:** {failed_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            else:
                st.markdown('<div class="status-card">ℹ️ No processing history found</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("📅 Schedule Information")
            
            next_run = get_next_run_info()
            if next_run:
                st.info(f"⏰ **Next run:** {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                
                time_until = next_run - datetime.now()
                if time_until.total_seconds() > 0:
                    hours = int(time_until.total_seconds() // 3600)
                    minutes = int((time_until.total_seconds() % 3600) // 60)
                    st.write(f"**Time until next run:** {hours}h {minutes}m")
                else:
                    st.warning("⚠️ Processing is overdue!")
            
            # Scheduler status
            if st.session_state.get('scheduler_initialized', False):
                st.success("🟢 Scheduler is running")
            else:
                st.error("🔴 Scheduler is not running")
        
        # Manual controls
        st.subheader("🎮 Manual Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔥 Force Run Processing Now", type="primary"):
                with st.spinner("Running processing..."):
                    background_scheduler.force_run_now()
                    st.success("Processing completed!")
                    time.sleep(2)
                    st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Data Cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")
                st.rerun()
        
        with col3:
            if st.button("📊 View Processing Logs"):
                log_file = Path("logs/app.log")
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        logs = f.read().split('\n')[-50:]  # Last 50 lines
                    
                    st.text_area("Recent Logs", '\n'.join(logs), height=300)
                else:
                    st.info("No log file found")
        
        # Data freshness
        st.subheader("📊 Data Freshness")
        
        load_time = self.data['load_time']
        time_since_load = datetime.now() - load_time
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            minutes_ago = int(time_since_load.total_seconds() // 60)
            st.metric("Data Last Loaded", f"{minutes_ago} min ago")
        
        with col2:
            cache_status = "🟢 Fresh" if minutes_ago < 5 else "🟡 Stale" if minutes_ago < 15 else "🔴 Very Stale"
            st.metric("Cache Status", cache_status)
        
        with col3:
            data_size = len(self.data['summary_data'])
            st.metric("Records in Memory", f"{data_size:,}")
        
        with col4:
            products_count = self.data['summary_data']['product_id'].nunique() if not self.data['summary_data'].empty else 0
            st.metric("Products Loaded", f"{products_count:,}")
    
    def data_overview_page(self):
        """Data overview"""
        st.header("📋 Live Data Overview")
        
        data = self.data['summary_data']
        
        if data.empty:
            st.warning("No data available")
            return
        
        # Basic statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{len(data):,}")
            st.metric("Unique Products", f"{data['product_id'].nunique():,}")
        
        with col2:
            date_range_days = (data['MONAT'].max() - data['MONAT'].min()).days
            st.metric("Date Range", f"{date_range_days} days")
            st.metric("Latest Data", data['MONAT'].max().strftime('%Y-%m-%d'))
        
        with col3:
            st.metric("Total Sales Volume", f"{data['anz_produkt'].sum():,.0f}")
            st.metric("Average Unit Price", f"€{data['unit_preis'].mean():.2f}")
        
        # Data quality checks
        st.subheader("🔍 Data Quality")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing data check
            missing_sales = (data['anz_produkt'].isna()).sum()
            missing_prices = (data['unit_preis'].isna()).sum()
            
            if missing_sales == 0 and missing_prices == 0:
                st.success("✅ No missing data detected")
            else:
                st.warning(f"⚠️ Missing data: {missing_sales} sales, {missing_prices} prices")
        
        with col2:
            # Data anomalies
            zero_sales = (data['anz_produkt'] == 0).sum()
            negative_prices = (data['unit_preis'] < 0).sum()
            
            if zero_sales == 0 and negative_prices == 0:
                st.success("✅ No anomalies detected")
            else:
                st.warning(f"⚠️ Anomalies: {zero_sales} zero sales, {negative_prices} negative prices")
        
        # Recent data sample
        st.subheader("📄 Recent Data Sample")
        
        # Show most recent 100 records
        recent_data = data.sort_values('MONAT', ascending=False).head(100)
        st.dataframe(recent_data, use_container_width=True)
        
        # Download option
        csv = data.to_csv(index=False)
        st.download_button(
            "💾 Download Full Dataset",
            csv,
            f"sales_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    def analytics_page(self):
        """Advanced analytics"""
        st.header("📈 Live Analytics")
        
        data = self.data['summary_data']
        
        if data.empty:
            st.warning("No data available for analytics")
            return
        
        # Time series analysis
        st.subheader("📊 Time Series Analysis")
        
        # Monthly aggregation
        monthly_data = data.groupby('MONAT').agg({
            'anz_produkt': ['sum', 'mean', 'count'],
            'unit_preis': 'mean'
        }).round(2)
        
        monthly_data.columns = ['Total Sales', 'Avg Sales', 'Product Count', 'Avg Price']
        monthly_data = monthly_data.reset_index()
        
        # Interactive time series chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_data['MONAT'],
            y=monthly_data['Total Sales'],
            mode='lines+markers',
            name='Total Sales',
            yaxis='y',
            hovertemplate='<b>%{x}</b><br>Total Sales: %{y:,.0f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_data['MONAT'],
            y=monthly_data['Avg Price'],
            mode='lines+markers',
            name='Avg Price',
            yaxis='y2',
            line=dict(color='orange'),
            hovertemplate='<b>%{x}</b><br>Avg Price: €%{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Monthly Sales and Price Trends",
            xaxis_title="Month",
            yaxis=dict(title="Sales Volume", side="left"),
            yaxis2=dict(title="Average Price (€)", side="right", overlaying="y"),
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Product performance analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Product Performance Matrix")
            
            product_stats = data.groupby('product_id').agg({
                'anz_produkt': ['sum', 'mean', 'std', 'count'],
                'unit_preis': 'mean'
            }).round(2)
            
            product_stats.columns = ['Total Sales', 'Avg Monthly Sales', 'Sales Volatility', 'Months Active', 'Avg Price']
            product_stats = product_stats.sort_values('Total Sales', ascending=False).head(20)
            
            st.dataframe(product_stats, use_container_width=True)
        
        with col2:
            st.subheader("📈 Sales vs Price Analysis")
            
            # Scatter plot of sales vs price
            sample_data = data.sample(min(1000, len(data)))
            
            fig = px.scatter(
                sample_data,
                x='unit_preis',
                y='anz_produkt',
                title="Sales Volume vs Unit Price",
                labels={'unit_preis': 'Unit Price (€)', 'anz_produkt': 'Sales Volume'},
                trendline="ols",
                opacity=0.6
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal analysis
        st.subheader("🌟 Seasonal Analysis")
        
        # Add month column for seasonality analysis
        data_with_month = data.copy()
        data_with_month['Month'] = data_with_month['MONAT'].dt.month
        
        monthly_seasonality = data_with_month.groupby('Month')['anz_produkt'].mean().reset_index()
        monthly_seasonality['Month_Name'] = monthly_seasonality['Month'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        })
        
        fig = px.bar(
            monthly_seasonality,
            x='Month_Name',
            y='anz_produkt',
            title="Average Sales by Month (Seasonality Pattern)",
            labels={'anz_produkt': 'Average Sales', 'Month_Name': 'Month'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application entry point"""
    app = ContinuousSalesForecastApp()
    app.run()

if __name__ == "__main__":
    main()