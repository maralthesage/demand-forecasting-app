"""
Data utility functions for sales forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import re
from datetime import datetime, timedelta

from utils.logger import get_logger

logger = get_logger(__name__)


def normalize_product_id(
    df: pd.DataFrame, product_col: str = "product_id"
) -> pd.DataFrame:
    """
    Normalize product IDs by stripping whitespace and ensuring consistent format

    Args:
        df: DataFrame with product IDs
        product_col: Name of product ID column

    Returns:
        DataFrame with normalized product IDs
    """
    df = df.copy()
    if product_col in df.columns:
        df[product_col] = df[product_col].astype(str).str.strip()
    return df


def create_lager_artikel_mapping(art_nr: str, groesse: str, farbe: str) -> str:
    """
    Create lager artikel number from components

    Args:
        art_nr: Article number
        groesse: Size
        farbe: Color

    Returns:
        Formatted lager artikel number
    """
    groesse = str(groesse).strip() if pd.notna(groesse) else ""
    farbe = str(farbe).strip() if pd.notna(farbe) else ""

    return f"{str(art_nr).ljust(8)}{groesse.ljust(4)}{farbe.ljust(2)}".strip()


def detect_covid_period(
    date_col: pd.Series, covid_start: str = "2020-01-01", covid_end: str = "2022-12-31"
) -> pd.Series:
    """
    Detect COVID period in date series

    Args:
        date_col: Date column
        covid_start: COVID period start date
        covid_end: COVID period end date

    Returns:
        Boolean series indicating COVID period
    """
    covid_start = pd.to_datetime(covid_start)
    covid_end = pd.to_datetime(covid_end)

    return (date_col >= covid_start) & (date_col <= covid_end)


def normalize_covid_sales(
    sales: pd.Series, is_covid: pd.Series, normalization_factor: float = 0.8
) -> pd.Series:
    """
    Normalize sales during COVID period

    Args:
        sales: Sales values
        is_covid: Boolean series indicating COVID period
        normalization_factor: Factor to normalize COVID sales

    Returns:
        Normalized sales series
    """
    normalized_sales = sales.copy()
    normalized_sales.loc[is_covid] = sales.loc[is_covid] / normalization_factor
    return normalized_sales


def detect_stock_outs(
    sales: pd.Series, stock_history: Optional[pd.Series] = None, threshold: float = 0.05
) -> pd.Series:
    """
    Detect potential stock-out periods based on sales patterns

    Args:
        sales: Sales data
        stock_history: Optional stock level history
        threshold: Threshold for stock-out detection (as fraction of average sales)

    Returns:
        Boolean series indicating potential stock-outs
    """
    if stock_history is not None:
        # Use actual stock data if available
        return stock_history <= 0

    # Heuristic: sales dropped to near zero when average is much higher
    rolling_avg = sales.rolling(window=3, center=True).mean()
    avg_sales = sales.mean()

    return (sales == 0) & (rolling_avg > avg_sales * threshold)


def detect_price_changes(price_series: pd.Series, threshold: float = 0.1) -> pd.Series:
    """
    Detect significant price changes

    Args:
        price_series: Price time series
        threshold: Minimum change threshold (as fraction)

    Returns:
        Boolean series indicating price changes
    """
    price_change = price_series.pct_change().abs()
    return price_change > threshold


def clean_categorical_data(df: pd.DataFrame, cat_columns: List[str]) -> pd.DataFrame:
    """
    Clean categorical data by standardizing formats

    Args:
        df: DataFrame with categorical columns
        cat_columns: List of categorical column names

    Returns:
        DataFrame with cleaned categorical data
    """
    df = df.copy()

    for col in cat_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df[col] = df[col].replace(["NAN", "NONE", ""], pd.NA)

    return df


def aggregate_sales_data(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str = "MONAT",
    sales_col: str = "MENGE",
    price_col: str = "PREIS",
) -> pd.DataFrame:
    """
    Aggregate sales data by specified grouping columns

    Args:
        df: Raw sales DataFrame
        group_cols: Columns to group by
        date_col: Date column name
        sales_col: Sales quantity column name
        price_col: Price column name

    Returns:
        Aggregated DataFrame
    """
    # Convert date to monthly periods
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp()

    # Group and aggregate
    agg_dict = {sales_col: "sum", price_col: "mean"}

    result = df.groupby(group_cols + [date_col]).agg(agg_dict).reset_index()
    result = result.rename(columns={sales_col: "anz_produkt", price_col: "unit_preis"})

    return result


def validate_data_quality(df: pd.DataFrame, required_cols: List[str]) -> Dict[str, any]:
    """
    Validate data quality and return quality metrics

    Args:
        df: DataFrame to validate
        required_cols: Required columns

    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        "total_rows": len(df),
        "missing_columns": [],
        "missing_data_pct": {},
        "duplicate_rows": 0,
        "data_types": {},
        "date_range": None,
    }

    # Check missing columns
    quality_report["missing_columns"] = [
        col for col in required_cols if col not in df.columns
    ]

    # Check missing data percentage
    for col in df.columns:
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        quality_report["missing_data_pct"][col] = round(missing_pct, 2)

    # Check duplicates
    quality_report["duplicate_rows"] = df.duplicated().sum()

    # Data types
    quality_report["data_types"] = df.dtypes.to_dict()

    # Date range (if date column exists)
    date_cols = [
        col for col in df.columns if "date" in col.lower() or "monat" in col.lower()
    ]
    if date_cols:
        date_col = date_cols[0]
        valid_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not valid_dates.empty:
            quality_report["date_range"] = {
                "start": valid_dates.min(),
                "end": valid_dates.max(),
                "span_months": (valid_dates.max() - valid_dates.min()).days // 30,
            }

    return quality_report


def filter_outliers(
    df: pd.DataFrame, column: str, method: str = "iqr", factor: float = 1.5
) -> pd.DataFrame:
    """
    Filter outliers from a DataFrame column

    Args:
        df: Input DataFrame
        column: Column to filter outliers from
        method: Method to use ('iqr' or 'zscore')
        factor: Outlier detection factor

    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()

    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)

    elif method == "zscore":
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        mask = z_scores <= factor

    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

    outliers_removed = len(df) - mask.sum()
    logger.info(
        f"Removed {outliers_removed} outliers from {column} using {method} method"
    )

    return df[mask]
