import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Superstore Full Dashboard", layout="wide")

# Load data
file_path = r"C:\Users\babo2\Downloads\Superstore Sales Dataset.csv"
df = pd.read_csv(file_path)

# Clean columns
df.columns = df.columns.str.strip().str.replace(" ", "_")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Column types
numeric_cols = df.select_dtypes(include=["float64", "int64"])
categorical_cols = df.select_dtypes(include=["object"])

# Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Summary Statistics")
st.dataframe(df.describe())

st.subheader("Missing Values")
st.write(df.isnull().sum())

# Detect key columns safely
sales_cols = [c for c in df.columns if 'sales' in c.lower()]
profit_cols = [c for c in df.columns if 'profit' in c.lower()]
quantity_cols = [c for c in df.columns if 'quantity' in c.lower()]

sales_col = sales_cols[0] if sales_cols else None
profit_col = profit_cols[0] if profit_cols else None
quantity_col = quantity_cols[0] if quantity_cols else None

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    if sales_col:
        st.metric("Total Sales", df[sales_col].sum())
    else:
        st.metric("Total Sales", 0)
with col2:
    if profit_col:
        st.metric("Total Profit", df[profit_col].sum())
    else:
        st.metric("Total Profit", 0)
with col3:
    if quantity_col:
        st.metric("Total Quantity", df[quantity_col].sum())
    else:
        st.metric("Total Quantity", 0)

# Order & Ship dates with error handling
order_cols = [c for c in df.columns if 'order_date' in c.lower()]
if order_cols:
    order_col = order_cols[0]
    df[order_col] = pd.to_datetime(df[order_col], errors='coerce')
    df['order_month'] = df[order_col].dt.to_period("M").astype(str)

ship_cols = [c for c in df.columns if 'ship_date' in c.lower()]
if ship_cols and order_cols:
    ship_col = ship_cols[0]
    df[ship_col] = pd.to_datetime(df[ship_col], errors='coerce')
    df['ship_delay_days'] = (df[ship_col] - df[order_col]).dt.days

# Calculated columns
if sales_col and profit_col:
    df['profit_margin'] = df[profit_col] / df[sales_col].replace({0: np.nan})
    df['profit_margin'] = df['profit_margin'].fillna(0)

if sales_col and quantity_col:
    df['price_per_unit'] = df[sales_col] / df[quantity_col].replace({0: np.nan})

# Outlier treatment
for c in numeric_cols.columns:
    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)
    IQR = Q3 - Q1
    low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df[c] = np.where(df[c]<low, low, df[c])
    df[c] = np.where(df[c]>high, high, df[c])

st.header("📊 Analysis Dashboard")

# Sales by Category
if 'Category' in df.columns and sales_col:
    st.subheader("Sales by Category")
    fig1 = px.bar(df, x="Category", y=sales_col, color="Category")
    st.plotly_chart(fig1, use_container_width=True)

# Sales by Region
if 'Region' in df.columns and sales_col:
    st.subheader("Sales by Region")
    fig2 = px.pie(df, names="Region", values=sales_col)
    st.plotly_chart(fig2, use_container_width=True)

# Profit by Sub-Category
if 'Sub-Category' in df.columns and profit_col:
    st.subheader("Profit by Sub-Category")
    fig3 = px.bar(df, x="Sub-Category", y=profit_col, color=profit_col)
    st.plotly_chart(fig3, use_container_width=True)

# Monthly Sales Trend
if 'order_month' in df.columns and sales_col:
    st.subheader("Monthly Sales Trend")
    ms = df.groupby("order_month")[sales_col].sum().reset_index()
    fig4 = px.line(ms, x="order_month", y=sales_col)
    st.plotly_chart(fig4, use_container_width=True)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig5 = px.imshow(df.select_dtypes(include=["float64","int64"]).corr())
st.plotly_chart(fig5, use_container_width=True)

st.success("✅ Full Analysis Completed")
