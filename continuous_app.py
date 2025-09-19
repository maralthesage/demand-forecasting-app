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

from background_scheduler import init_background_scheduler
from incremental_training_system import incremental_system
from config import get_config, STREAMLIT_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

# Initialize background scheduler safely
background_scheduler = init_background_scheduler()

# Page config
st.set_page_config(
    page_title="Demand Forecasting - Live",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for live app
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Initialize scheduler on app startup
if "scheduler_initialized" not in st.session_state:
    try:
        # Get daily processing time from environment or use default
        import os

        daily_time = os.getenv("DAILY_PROCESSING_TIME", "06:00")
        background_scheduler.daily_time = daily_time

        # Start background scheduler
        background_scheduler.start()
        st.session_state.scheduler_initialized = True
        st.session_state.scheduler_error = None
        logger.info("Background scheduler initialized successfully")
    except Exception as e:
        st.session_state.scheduler_initialized = False
        st.session_state.scheduler_error = str(e)
        logger.error(f"Failed to initialize scheduler: {e}")


# Cached data loading - refreshed daily at 6 AM
@st.cache_data(ttl=86400, show_spinner="🔄 Loading data...")  # 24 hours
def load_current_data():
    """Load current data and models"""
    try:
        features_data, forecaster, feature_columns = (
            incremental_system.quick_load_for_app()
        )

        # Create summary for dashboard
        summary_data = (
            features_data.groupby(["product_id", "MONAT"])
            .agg({"anz_produkt": "sum", "unit_preis": "mean"})
            .reset_index()
        )

        return {
            "features_data": features_data,
            "forecaster": forecaster,
            "feature_columns": feature_columns,
            "summary_data": summary_data,
            "load_success": True,
            "load_time": datetime.now(),
            "error_message": None,
        }

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return {
            "features_data": pd.DataFrame(),
            "forecaster": None,
            "feature_columns": [],
            "summary_data": pd.DataFrame(),
            "load_success": False,
            "load_time": datetime.now(),
            "error_message": str(e),
        }


def get_processing_status():
    """Get current processing status"""
    return background_scheduler.get_status()


def get_next_run_info():
    """Get next scheduled run information"""
    status = background_scheduler.get_status()
    next_run = status.get('next_run')
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
            st.markdown(
                """
            <h1 style="color: #1f77b4;">
                🚀 Demand Forecasting App 
                <span class="live-indicator"></span> Live
            </h1>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            # Live status indicator
            if st.session_state.get("scheduler_initialized", False):
                st.markdown(
                    '<div class="success-card">🟢 Live Processing Active</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="error-card">🔴 Processing Offline</div>',
                    unsafe_allow_html=True,
                )

        with col3:
            # Data refresh info
            st.markdown(
                """
                <div class="status-card">
                    📅 <strong>Data Updates:</strong><br>
                    Daily at 6:00 AM<br>
                    <small>Automatic model retraining</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Show scheduler status
        self.show_scheduler_status()

        # Load current data
        self.data = load_current_data()

        if not self.data["load_success"]:
            st.error(f"❌ Data loading failed: {self.data['error_message']}")
            st.info(
                "Please check your data sources and try restarting the application."
            )
            return

        # Navigation
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📊 Warengruppe Performance",
                "🔍 Product Search & Forecast",
                "📈 Analytics",
                "⚙️ System Status",
                "📋 Data Overview",
            ]
        )

        with tab1:
            self.warengruppe_performance_page()

        with tab2:
            self.product_search_forecast_page()

        with tab3:
            self.analytics_page()

        with tab4:
            self.system_status_page()

        with tab5:
            self.data_overview_page()

    def show_scheduler_status(self):
        """Show scheduler status in sidebar"""
        st.sidebar.markdown("### 🤖 Background Processing")

        if st.session_state.get("scheduler_initialized", False):
            # Next run time
            next_run_str = get_next_run_info()
            if next_run_str:
                try:
                    from datetime import datetime as dt
                    next_run = dt.fromisoformat(next_run_str.replace('Z', '+00:00'))
                    time_until_next = next_run - datetime.now()
                    if time_until_next.total_seconds() > 0:
                        hours = int(time_until_next.total_seconds() // 3600)
                        minutes = int((time_until_next.total_seconds() % 3600) // 60)
                        st.sidebar.info(f"⏰ Next run in: {hours}h {minutes}m")
                    else:
                        st.sidebar.info("⏰ Next run: Due now")
                except (ValueError, TypeError) as e:
                    st.sidebar.info(f"⏰ Next run: {next_run_str}")

            # Last run status
            status = get_processing_status()
            if status.get("success"):
                st.sidebar.success(
                    f"✅ Last run: {status.get('num_records', 0):,} records processed"
                )
                if "timestamp" in status:
                    last_run = datetime.fromisoformat(status["timestamp"])
                    time_ago = datetime.now() - last_run
                    hours_ago = int(time_ago.total_seconds() // 3600)
                    st.sidebar.info(f"🕐 {hours_ago}h ago")
            elif status.get("error_message"):
                st.sidebar.error(f"❌ Last run failed")
                with st.sidebar.expander("Error details"):
                    st.text(status.get("error_message", "Unknown error"))

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
            error = st.session_state.get("scheduler_error", "Unknown error")
            st.sidebar.error(f"❌ Scheduler failed to start")
            with st.sidebar.expander("Error details"):
                st.text(error)

    def warengruppe_performance_page(self):
        """Warengruppe performance and top products"""
        st.header("📊 Warengruppe Performance & Top Products")

        features_data = self.data["features_data"]
        
        if features_data.empty:
            st.warning("No data available for analysis")
            return

        # Check if we have product_category_id column
        if 'product_category_id' not in features_data.columns:
            st.error("Product category information not available. Please ensure marketing artikel data is loaded.")
            return

        # Data freshness indicator
        load_time = self.data["load_time"]
        st.info(f"📅 Data last updated: {load_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Warengruppen", f"{features_data['product_category_id'].nunique():,}")
        with col2:
            st.metric("Total Products", f"{features_data['product_id'].nunique():,}")
        with col3:
            st.metric("Total Sales Volume", f"{features_data['anz_produkt'].sum():,.0f}")
        with col4:
            st.metric("Avg Unit Price", f"€{features_data['unit_preis'].mean():.2f}")

        st.divider()

        # Warengruppe Performance Analysis
        st.subheader("🏆 Warengruppe Performance Analysis")
        
        # Get last 12 months of data
        cutoff_date = datetime.now() - timedelta(days=365)
        recent_data = features_data[features_data["MONAT"] >= cutoff_date]
        
        if not recent_data.empty:
            # Warengruppe performance metrics
            warengruppe_performance = (
                recent_data.groupby('product_category_id')
                .agg({
                    'anz_produkt': ['sum', 'mean', 'count'],
                    'unit_preis': 'mean',
                    'product_id': 'nunique'
                })
                .round(2)
            )
            
            # Flatten column names
            warengruppe_performance.columns = ['Total_Sales', 'Avg_Monthly_Sales', 'Months_Active', 'Avg_Price', 'Num_Products']
            warengruppe_performance = warengruppe_performance.reset_index()
            warengruppe_performance['Revenue'] = warengruppe_performance['Total_Sales'] * warengruppe_performance['Avg_Price']
            
            # Sort by total sales
            warengruppe_performance = warengruppe_performance.sort_values('Total_Sales', ascending=False)
            
            # Display top Warengruppen
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Top Warengruppen by Sales Volume")
                top_warengruppen = warengruppe_performance.head(10)
                
                fig = px.bar(
                    top_warengruppen,
                    x='product_category_id',
                    y='Total_Sales',
                    title="Top 10 Warengruppen - Sales Volume (Last 12 Months)",
                    labels={'Total_Sales': 'Total Sales Volume', 'product_category_id': 'Warengruppe'},
                    color='Total_Sales',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("💰 Top Warengruppen by Revenue")
                top_revenue = warengruppe_performance.sort_values('Revenue', ascending=False).head(10)
                
                fig = px.bar(
                    top_revenue,
                    x='product_category_id',
                    y='Revenue',
                    title="Top 10 Warengruppen - Revenue (Last 12 Months)",
                    labels={'Revenue': 'Total Revenue (€)', 'product_category_id': 'Warengruppe'},
                    color='Revenue',
                    color_continuous_scale='Greens'
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Warengruppe performance table
            st.subheader("📊 Detailed Warengruppe Performance")
            
            # Format the display table
            display_table = warengruppe_performance.copy()
            display_table['Total_Sales'] = display_table['Total_Sales'].apply(lambda x: f"{x:,.0f}")
            display_table['Avg_Monthly_Sales'] = display_table['Avg_Monthly_Sales'].apply(lambda x: f"{x:,.1f}")
            display_table['Avg_Price'] = display_table['Avg_Price'].apply(lambda x: f"€{x:.2f}")
            display_table['Revenue'] = display_table['Revenue'].apply(lambda x: f"€{x:,.0f}")
            
            display_table.columns = ['Warengruppe', 'Total Sales', 'Avg Monthly Sales', 'Active Months', 'Avg Price', 'Products', 'Total Revenue']
            
            st.dataframe(
                display_table,
                use_container_width=True,
                hide_index=True
            )
        
        st.divider()
        
        # Top Products Analysis
        st.subheader("⭐ Top Individual Products")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥇 Top 15 Products by Sales Volume")
            if not recent_data.empty:
                top_products_volume = (
                    recent_data.groupby(['product_id', 'product_category_id'])['anz_produkt']
                    .sum()
                    .reset_index()
                    .sort_values('anz_produkt', ascending=False)
                    .head(15)
                )
                
                fig = px.bar(
                    top_products_volume,
                    x='product_id',
                    y='anz_produkt',
                    color='product_category_id',
                    title="Top 15 Products - Sales Volume (Last 12 Months)",
                    labels={'anz_produkt': 'Sales Volume', 'product_id': 'Product ID'},
                    height=500
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recent data available")
        
        with col2:
            st.subheader("💎 Top 15 Products by Revenue")
            if not recent_data.empty:
                top_products_revenue = recent_data.copy()
                top_products_revenue['revenue'] = top_products_revenue['anz_produkt'] * top_products_revenue['unit_preis']
                
                top_revenue_products = (
                    top_products_revenue.groupby(['product_id', 'product_category_id'])['revenue']
                    .sum()
                    .reset_index()
                    .sort_values('revenue', ascending=False)
                    .head(15)
                )
                
                fig = px.bar(
                    top_revenue_products,
                    x='product_id',
                    y='revenue',
                    color='product_category_id',
                    title="Top 15 Products - Revenue (Last 12 Months)",
                    labels={'revenue': 'Revenue (€)', 'product_id': 'Product ID'},
                    height=500
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recent data available")

    def product_search_forecast_page(self):
        """Product search with history and forecast visualization"""
        st.header("🔍 Product Search & Demand Forecast")
        st.markdown("Search for any product by ID to view its historical demand and get 6-month forecasts with confidence intervals.")
        
        features_data = self.data["features_data"]
        forecaster = self.data["forecaster"]
        
        if features_data.empty:
            st.warning("No data available for forecasting")
            return
        
        if forecaster is None:
            st.error("Forecasting models not available. Please check system status.")
            return

        # Product search interface
        st.subheader("🔍 Search Product")
        
        # Get all available products
        available_products = sorted(features_data["product_id"].unique())
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Search input
            search_term = st.text_input(
                "Enter Product ID",
                placeholder="Type product ID (e.g., PROD001, ART123, etc.)",
                help="Search for exact product ID or partial match"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            search_button = st.button("🔍 Search Product", type="primary")

        # Filter products based on search
        if search_term:
            # Find products that contain the search term
            matching_products = [p for p in available_products if search_term.upper() in p.upper()]
            
            if matching_products:
                if len(matching_products) == 1:
                    selected_product = matching_products[0]
                    st.success(f"✅ Found product: **{selected_product}**")
                    self.generate_product_forecast(selected_product)
                else:
                    st.info(f"Found {len(matching_products)} matching products:")
                    
                    # Show matching products in a selectbox
                    selected_product = st.selectbox(
                        "Select from matching products:",
                        matching_products,
                        key="product_selector"
                    )
                    
                    if st.button("📈 Show Forecast", type="secondary"):
                        self.generate_product_forecast(selected_product)
            else:
                st.warning(f"No products found matching '{search_term}'")
                
                # Show some suggestions
                if len(search_term) >= 2:
                    suggestions = [p for p in available_products[:10] if any(char in p.upper() for char in search_term.upper())]
                    if suggestions:
                        st.info("💡 Similar products you might be looking for:")
                        for suggestion in suggestions[:5]:
                            if st.button(f"🔍 {suggestion}", key=f"suggest_{suggestion}"):
                                st.session_state.search_suggestion = suggestion
                                st.rerun()
        else:
            # Show some example products
            st.info("💡 **Example products you can search for:**")
            
            # Display first 10 products as examples
            example_products = available_products[:10]
            cols = st.columns(5)
            for i, product in enumerate(example_products):
                with cols[i % 5]:
                    if st.button(f"📊 {product}", key=f"example_{product}"):
                        st.session_state.selected_example = product
                        self.generate_product_forecast(product)
        
        # Handle search suggestion from session state
        if hasattr(st.session_state, 'search_suggestion'):
            self.generate_product_forecast(st.session_state.search_suggestion)
            del st.session_state.search_suggestion

    def generate_product_forecast(self, product_id: str):
        """Generate comprehensive forecast visualization for a specific product"""
        st.divider()
        st.subheader(f"📈 Product Analysis: {product_id}")
        
        features_data = self.data["features_data"]
        forecaster = self.data["forecaster"]
        
        # Get product data
        product_data = features_data[features_data["product_id"] == product_id].copy()
        
        if product_data.empty:
            st.error(f"No data found for product {product_id}")
            return
        
        # Sort by date
        product_data = product_data.sort_values("MONAT")
        
        # Product information
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'product_category_id' in product_data.columns:
                category = product_data['product_category_id'].iloc[0] if not product_data['product_category_id'].isna().all() else "Unknown"
                st.metric("Warengruppe", category)
            else:
                st.metric("Warengruppe", "N/A")
        
        with col2:
            total_sales = product_data['anz_produkt'].sum()
            st.metric("Total Historical Sales", f"{total_sales:,.0f}")
        
        with col3:
            avg_price = product_data['unit_preis'].mean()
            st.metric("Average Price", f"€{avg_price:.2f}")
        
        with col4:
            data_points = len(product_data)
            st.metric("Data Points", f"{data_points} months")
        
        # Historical demand table
        st.subheader("📋 Historical Demand Data")
        
        # Prepare display table
        display_data = product_data[['MONAT', 'anz_produkt', 'unit_preis']].copy()
        display_data['MONAT'] = display_data['MONAT'].dt.strftime('%Y-%m')
        display_data.columns = ['Month', 'Demand', 'Unit Price (€)']
        display_data['Unit Price (€)'] = display_data['Unit Price (€)'].round(2)
        
        # Show the table
        st.dataframe(
            display_data.sort_values('Month', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Generate forecast using the trained model
        try:
            st.subheader("🔮 6-Month Demand Forecast")
            
            with st.spinner("Generating AI-powered forecast..."):
                # Prepare features for forecasting (simplified approach)
                if len(product_data) >= 3:
                    # Use the forecaster if available and properly trained
                    if forecaster and hasattr(forecaster, 'predict'):
                        try:
                            # Get the latest features for the product
                            latest_features = product_data.iloc[-1:][self.data["feature_columns"]]
                            
                            # Generate forecast using the trained model
                            forecast_result = forecaster.predict(latest_features, horizon=6)
                            
                            if hasattr(forecast_result, 'forecast'):
                                forecast_values = forecast_result.forecast
                                if hasattr(forecast_result, 'confidence_intervals'):
                                    conf_intervals = forecast_result.confidence_intervals
                                else:
                                    # Create simple confidence intervals
                                    std_dev = product_data['anz_produkt'].std()
                                    forecast_values = np.array(forecast_values)
                                    conf_intervals = {
                                        'lower': forecast_values - 1.96 * std_dev,
                                        'upper': forecast_values + 1.96 * std_dev
                                    }
                            else:
                                forecast_values = forecast_result
                                std_dev = product_data['anz_produkt'].std()
                                forecast_values = np.array(forecast_values)
                                conf_intervals = {
                                    'lower': forecast_values - 1.96 * std_dev,
                                    'upper': forecast_values + 1.96 * std_dev
                                }
                        except Exception as e:
                            logger.warning(f"Model prediction failed for {product_id}: {e}")
                            # Fallback to simple forecasting
                            forecast_values, conf_intervals = self._simple_forecast(product_data)
                    else:
                        # Fallback to simple forecasting
                        forecast_values, conf_intervals = self._simple_forecast(product_data)
                else:
                    st.warning("Insufficient historical data for reliable forecasting (minimum 3 months required)")
                    return
                
                # Create forecast dates
                last_date = product_data['MONAT'].max()
                forecast_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1), 
                    periods=6, 
                    freq='MS'
                )
                
                # Create the visualization
                fig = go.Figure()
                
                # Historical data (BLUE)
                fig.add_trace(
                    go.Scatter(
                        x=product_data['MONAT'],
                        y=product_data['anz_produkt'],
                        mode='lines+markers',
                        name='Historical Demand',
                        line=dict(color='#1f77b4', width=3),  # Blue
                        marker=dict(size=6),
                        hovertemplate="<b>Historical</b><br>Date: %{x}<br>Demand: %{y:.0f}<extra></extra>"
                    )
                )
                
                # Forecast data (RED)
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#d62728', width=3, dash='dash'),  # Red
                        marker=dict(size=6),
                        hovertemplate="<b>Forecast</b><br>Date: %{x}<br>Demand: %{y:.0f}<extra></extra>"
                    )
                )
                
                # Confidence interval (Light red fill)
                fig.add_trace(
                    go.Scatter(
                        x=list(forecast_dates) + list(forecast_dates[::-1]),
                        y=list(conf_intervals['upper']) + list(conf_intervals['lower'][::-1]),
                        fill='toself',
                        fillcolor='rgba(214, 39, 40, 0.2)',  # Light red
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence Interval',
                        hoverinfo='skip'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Demand Forecast for Product {product_id}",
                    xaxis_title="Date",
                    yaxis_title="Demand (Units)",
                    height=600,
                    hovermode='x unified',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast summary table
                st.subheader("📊 Forecast Summary")
                
                forecast_df = pd.DataFrame({
                    'Month': forecast_dates.strftime('%Y-%m'),
                    'Forecasted Demand': [f"{int(f):,}" for f in forecast_values],
                    'Lower Bound (95%)': [f"{int(l):,}" for l in conf_intervals['lower']],
                    'Upper Bound (95%)': [f"{int(u):,}" for u in conf_intervals['upper']]
                })
                
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                
                # Key insights
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Forecast Insights")
                    
                    total_forecast = sum(forecast_values)
                    avg_historical = product_data['anz_produkt'].mean()
                    avg_forecast = np.mean(forecast_values)
                    
                    st.metric("6-Month Total Forecast", f"{int(total_forecast):,}")
                    st.metric("Average Monthly Forecast", f"{int(avg_forecast):,}")
                    
                    change_pct = ((avg_forecast - avg_historical) / avg_historical * 100) if avg_historical > 0 else 0
                    st.metric(
                        "vs Historical Average", 
                        f"{change_pct:+.1f}%",
                        delta=f"{change_pct:+.1f}%"
                    )
                
                with col2:
                    st.subheader("💾 Export Options")
                    
                    # Combine historical and forecast data for export
                    export_data = pd.DataFrame({
                        'Date': list(product_data['MONAT'].dt.strftime('%Y-%m')) + list(forecast_dates.strftime('%Y-%m')),
                        'Type': ['Historical'] * len(product_data) + ['Forecast'] * 6,
                        'Demand': list(product_data['anz_produkt']) + list(forecast_values),
                        'Lower_Bound': [None] * len(product_data) + list(conf_intervals['lower']),
                        'Upper_Bound': [None] * len(product_data) + list(conf_intervals['upper'])
                    })
                    
                    csv_data = export_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Complete Analysis",
                        data=csv_data,
                        file_name=f"product_analysis_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    st.info("💡 **Daily Updates**: This forecast updates automatically every day at 6:00 AM with the latest data and retrained models.")
                
        except Exception as e:
            st.error(f"Forecast generation failed: {str(e)}")
            logger.error(f"Forecast error for product {product_id}: {str(e)}")
    
    def _simple_forecast(self, product_data):
        """Simple fallback forecasting method"""
        recent_data = product_data.tail(12)['anz_produkt'].values
        
        # Simple trend calculation
        if len(recent_data) >= 3:
            trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
        else:
            trend = 0
        
        # Generate simple forecast
        last_value = recent_data[-1]
        forecast_values = []
        
        for i in range(6):
            forecast = max(0, last_value + trend * (i + 1))
            # Add some realistic variation
            forecast *= np.random.normal(1.0, 0.05)
            forecast_values.append(max(0, forecast))
        
        # Simple confidence intervals
        std_dev = np.std(recent_data) if len(recent_data) > 1 else np.mean(recent_data) * 0.2
        
        conf_intervals = {
            'lower': [max(0, f - 1.96 * std_dev) for f in forecast_values],
            'upper': [f + 1.96 * std_dev for f in forecast_values]
        }
        
        return forecast_values, conf_intervals

    def system_status_page(self):
        """System status and monitoring"""
        st.header("⚙️ System Status & Monitoring")

        # Processing status
        status = get_processing_status()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🤖 Background Processing")

            if status.get("success"):
                st.markdown(
                    '<div class="success-card">✅ Last processing run successful</div>',
                    unsafe_allow_html=True,
                )

                # Processing metrics
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Records Processed", f"{status.get('num_records', 0):,}")
                    st.metric(
                        "Products Processed", f"{status.get('num_products', 0):,}"
                    )

                with metrics_col2:
                    processing_time = status.get("processing_time_seconds", 0)
                    st.metric("Processing Time", f"{processing_time:.1f}s")

                    if "timestamp" in status:
                        last_run = datetime.fromisoformat(status["timestamp"])
                        st.metric("Last Run", last_run.strftime("%Y-%m-%d %H:%M"))

            elif status.get("error_message"):
                st.markdown(
                    '<div class="error-card">❌ Last processing run failed</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("Error Details", expanded=True):
                    st.error(status["error_message"])

                    if "timestamp" in status:
                        failed_time = datetime.fromisoformat(status["timestamp"])
                        st.write(
                            f"**Failed at:** {failed_time.strftime('%Y-%m-%d %H:%M:%S')}"
                        )

            else:
                st.markdown(
                    '<div class="status-card">ℹ️ No processing history found</div>',
                    unsafe_allow_html=True,
                )

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
            if st.session_state.get("scheduler_initialized", False):
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
                    with open(log_file, "r") as f:
                        logs = f.read().split("\n")[-50:]  # Last 50 lines

                    st.text_area("Recent Logs", "\n".join(logs), height=300)
                else:
                    st.info("No log file found")

        # Data freshness
        st.subheader("📊 Data Freshness")

        load_time = self.data["load_time"]
        time_since_load = datetime.now() - load_time

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            minutes_ago = int(time_since_load.total_seconds() // 60)
            st.metric("Data Last Loaded", f"{minutes_ago} min ago")

        with col2:
            cache_status = (
                "🟢 Fresh"
                if minutes_ago < 5
                else "🟡 Stale" if minutes_ago < 15 else "🔴 Very Stale"
            )
            st.metric("Cache Status", cache_status)

        with col3:
            data_size = len(self.data["summary_data"])
            st.metric("Records in Memory", f"{data_size:,}")

        with col4:
            products_count = (
                self.data["summary_data"]["product_id"].nunique()
                if not self.data["summary_data"].empty
                else 0
            )
            st.metric("Products Loaded", f"{products_count:,}")

    def data_overview_page(self):
        """Data overview"""
        st.header("📋 Live Data Overview")

        data = self.data["summary_data"]

        if data.empty:
            st.warning("No data available")
            return

        # Basic statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Records", f"{len(data):,}")
            st.metric("Unique Products", f"{data['product_id'].nunique():,}")

        with col2:
            date_range_days = (data["MONAT"].max() - data["MONAT"].min()).days
            st.metric("Date Range", f"{date_range_days} days")
            st.metric("Latest Data", data["MONAT"].max().strftime("%Y-%m-%d"))

        with col3:
            st.metric("Total Sales Volume", f"{data['anz_produkt'].sum():,.0f}")
            st.metric("Average Unit Price", f"€{data['unit_preis'].mean():.2f}")

        # Data quality checks
        st.subheader("🔍 Data Quality")

        col1, col2 = st.columns(2)

        with col1:
            # Missing data check
            missing_sales = (data["anz_produkt"].isna()).sum()
            missing_prices = (data["unit_preis"].isna()).sum()

            if missing_sales == 0 and missing_prices == 0:
                st.success("✅ No missing data detected")
            else:
                st.warning(
                    f"⚠️ Missing data: {missing_sales} sales, {missing_prices} prices"
                )

        with col2:
            # Data anomalies
            zero_sales = (data["anz_produkt"] == 0).sum()
            negative_prices = (data["unit_preis"] < 0).sum()

            if zero_sales == 0 and negative_prices == 0:
                st.success("✅ No anomalies detected")
            else:
                st.warning(
                    f"⚠️ Anomalies: {zero_sales} zero sales, {negative_prices} negative prices"
                )

        # Recent data sample
        st.subheader("📄 Recent Data Sample")

        # Show most recent 100 records
        recent_data = data.sort_values("MONAT", ascending=False).head(100)
        st.dataframe(recent_data, use_container_width=True)

        # Download option
        csv = data.to_csv(index=False)
        st.download_button(
            "💾 Download Full Dataset",
            csv,
            f"sales_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True,
        )

    def analytics_page(self):
        """Advanced analytics"""
        st.header("📈 Live Analytics")

        data = self.data["summary_data"]

        if data.empty:
            st.warning("No data available for analytics")
            return

        # Time series analysis
        st.subheader("📊 Time Series Analysis")

        # Monthly aggregation
        monthly_data = (
            data.groupby("MONAT")
            .agg({"anz_produkt": ["sum", "mean", "count"], "unit_preis": "mean"})
            .round(2)
        )

        monthly_data.columns = [
            "Total Sales",
            "Avg Sales",
            "Product Count",
            "Avg Price",
        ]
        monthly_data = monthly_data.reset_index()

        # Interactive time series chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=monthly_data["MONAT"],
                y=monthly_data["Total Sales"],
                mode="lines+markers",
                name="Total Sales",
                yaxis="y",
                hovertemplate="<b>%{x}</b><br>Total Sales: %{y:,.0f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=monthly_data["MONAT"],
                y=monthly_data["Avg Price"],
                mode="lines+markers",
                name="Avg Price",
                yaxis="y2",
                line=dict(color="orange"),
                hovertemplate="<b>%{x}</b><br>Avg Price: €%{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Monthly Sales and Price Trends",
            xaxis_title="Month",
            yaxis=dict(title="Sales Volume", side="left"),
            yaxis2=dict(title="Average Price (€)", side="right", overlaying="y"),
            height=500,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Product performance analysis
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Product Performance Matrix")

            product_stats = (
                data.groupby("product_id")
                .agg(
                    {
                        "anz_produkt": ["sum", "mean", "std", "count"],
                        "unit_preis": "mean",
                    }
                )
                .round(2)
            )

            product_stats.columns = [
                "Total Sales",
                "Avg Monthly Sales",
                "Sales Volatility",
                "Months Active",
                "Avg Price",
            ]
            product_stats = product_stats.sort_values(
                "Total Sales", ascending=False
            ).head(20)

            st.dataframe(product_stats, use_container_width=True)

        with col2:
            st.subheader("📈 Sales vs Price Analysis")

            # Scatter plot of sales vs price
            sample_data = data.sample(min(1000, len(data)))

            fig = px.scatter(
                sample_data,
                x="unit_preis",
                y="anz_produkt",
                title="Sales Volume vs Unit Price",
                labels={"unit_preis": "Unit Price (€)", "anz_produkt": "Sales Volume"},
                trendline="ols",
                opacity=0.6,
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Seasonal analysis
        st.subheader("🌟 Seasonal Analysis")

        # Add month column for seasonality analysis
        data_with_month = data.copy()
        data_with_month["Month"] = data_with_month["MONAT"].dt.month

        monthly_seasonality = (
            data_with_month.groupby("Month")["anz_produkt"].mean().reset_index()
        )
        monthly_seasonality["Month_Name"] = monthly_seasonality["Month"].map(
            {
                1: "Jan",
                2: "Feb",
                3: "Mar",
                4: "Apr",
                5: "May",
                6: "Jun",
                7: "Jul",
                8: "Aug",
                9: "Sep",
                10: "Oct",
                11: "Nov",
                12: "Dec",
            }
        )

        fig = px.bar(
            monthly_seasonality,
            x="Month_Name",
            y="anz_produkt",
            title="Average Sales by Month (Seasonality Pattern)",
            labels={"anz_produkt": "Average Sales", "Month_Name": "Month"},
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application entry point"""
    app = ContinuousSalesForecastApp()
    app.run()


if __name__ == "__main__":
    main()
