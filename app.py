import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector  # for SQL connection
from mysql.connector import Error

st.set_page_config(
    page_title="Online Food Delivery Analytics Dashboard",
    layout="wide"
)

st.title("🍔 Online Food Delivery Dashboard")
st.write("End-to-End Data Analysis Project using Python & Streamlit")

# ==============================
# Load Data: CSV or SQL
# ===============================

# Option 1: Load from CSV
df = pd.read_csv("processed_food_orders.csv")
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# Option 2: Load from SQL
@st.cache_data
def load_data_from_sql(query, host, user, password, database):
    try:
        connection = mysql.connector.connect(
            host= 'localhost',
            user= 'root',
            password= 'RoyFrank',
            database=database
        )
        df = pd.read_sql(query, connection)
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        return df
    except Error as e:
        st.error(f"Error connecting to database: {e}")
        return pd.DataFrame()
    finally:
        if connection.is_connected():
            connection.close()

# Example usage:
# query = "SELECT * FROM food_orders_analysis"
# df = load_data_from_sql(query, host="localhost", user="root", password="your_password", database="food_db")

# ===============================
# Sidebar Filters
# ===============================
st.sidebar.header("📌 Filters")

if 'df' in locals():
    start_date = st.sidebar.date_input("Start Date", df['Order_Date'].min())
    end_date = st.sidebar.date_input("End Date", df['Order_Date'].max())

    filtered_df = df[
        (df['Order_Date'] >= pd.to_datetime(start_date)) &
        (df['Order_Date'] <= pd.to_datetime(end_date))
    ]
else:
    st.warning("No data loaded.")
    filtered_df = pd.DataFrame()

# ===============================
# KPI Calculations
# ===============================
if not filtered_df.empty:
    total_orders = filtered_df['Order_ID'].nunique()
    total_revenue = filtered_df['Final_Amount'].sum()
    avg_order_value = filtered_df['Final_Amount'].mean()
    avg_delivery_time = filtered_df['Delivery_Time_Min'].mean()
    cancellation_rate = (filtered_df['Order_Status'] == 'Cancelled').mean() * 100
    avg_rating = filtered_df['Delivery_Rating'].mean()
    profit_margin = ((filtered_df['Final_Amount'] - filtered_df['Discount_Applied']).sum() / filtered_df['Final_Amount'].sum()) * 100

    # ===============================
    # KPI Section
    # ===============================
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", total_orders)
    col2.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    col3.metric("Avg Order Value", f"₹{avg_order_value:.2f}")
    col4.metric("Avg Delivery Time", f"{avg_delivery_time:.2f} mins")

    col5, col6, col7 = st.columns(3)
    col5.metric("Cancellation Rate", f"{cancellation_rate:.2f}%")
    col6.metric("Avg Delivery Rating", f"{avg_rating:.2f}")
    col7.metric("Profit Margin", f"{profit_margin:.2f}%")

    st.markdown("---")

    # ===============================
    # Total Orders
    # ===============================
    st.subheader("📈 Orders Trend Over Time")
    orders_trend = filtered_df.groupby(filtered_df['Order_Date'].dt.date).size()
    st.line_chart(orders_trend)

    # ===============================
    # Total Revenue
    # ===============================
    st.subheader("💰 Revenue Trend Over Time")

    # Convert Order_Date to datetime
    filtered_df['Order_Date'] = pd.to_datetime(filtered_df['Order_Date'])

    # Monthly revenue
    monthly_revenue = filtered_df.resample('M', on='Order_Date')['Final_Amount'].sum()

    # Display the chart
    st.bar_chart(monthly_revenue)
    st.markdown("---")

    # ===============================
    # Cancellation Rate
    # ===============================
    st.subheader("❌ Cancellation Analysis")
    cancel_counts = filtered_df['Order_Status'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(cancel_counts.index, cancel_counts.values, color='salmon')
    ax1.set_xlabel("Order Status")
    ax1.set_ylabel("Number of Orders")
    st.pyplot(fig1)

    # ===============================
    # Average Delivery Rating
    # ===============================
    st.subheader("⭐ Delivery Rating Distribution")
    rating_counts = filtered_df['Delivery_Rating'].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    ax2.bar(rating_counts.index, rating_counts.values, color='gold')
    ax2.set_xlabel("Rating")
    ax2.set_ylabel("Orders")
    st.pyplot(fig2)

    st.markdown("---")

    # =============================
    # Profit Margin Over Time
    # =============================
    st.subheader("💹 Profit Margin Over Time (Smoothed Daily Average)")

    # Calculate daily average profit margin
    daily_profit_margin = filtered_df.groupby(filtered_df['Order_Date'].dt.date).apply(
    lambda x: ((x['Final_Amount'] - x['Discount_Applied']).sum() / x['Final_Amount'].sum()) * 100
    )

    # Apply rolling average for smoothing (14-day window)
    daily_profit_margin_smooth = daily_profit_margin.rolling(window=14, min_periods=1).mean()

    fig_pm, ax_pm = plt.subplots(figsize=(12, 5))
    # Plot raw data
    ax_pm.plot(daily_profit_margin.index, daily_profit_margin.values, color='lightgreen', alpha=0.5, label='Daily Profit Margin')
    # Plot smoothed data
    ax_pm.plot(daily_profit_margin_smooth.index, daily_profit_margin_smooth.values, color='green', linewidth=2, label='Smoothed (14-day avg)')

    ax_pm.set_xlabel("Date")
    ax_pm.set_ylabel("Profit Margin (%)")
    ax_pm.set_title("Profit Margin Over Time")
    ax_pm.grid(True)
    ax_pm.legend()
    fig_pm.autofmt_xdate(rotation=45)
    st.pyplot(fig_pm)

    # ==============================
    # Average Order Value Over Time
    # ==============================
    st.subheader("💹 Average Order Value Over Time (Smoothed Daily Average)")

    # Calculate daily average order value
    daily_avg_order_value = filtered_df.groupby(filtered_df['Order_Date'].dt.date)['Final_Amount'].mean()

    # Apply rolling average for smoothing (14-day window)
    daily_avg_order_value_smooth = daily_avg_order_value.rolling(window=14, min_periods=1).mean()

    fig_aov, ax_aov = plt.subplots(figsize=(12, 5))
    # Plot raw data
    ax_aov.plot(daily_avg_order_value.index, daily_avg_order_value.values, color='lightblue', alpha=0.5, label='Daily Avg Order Value')
    # Plot smoothed data
    ax_aov.plot(daily_avg_order_value_smooth.index, daily_avg_order_value_smooth.values, color='blue', linewidth=2, label='Smoothed (14-day avg)')

    ax_aov.set_xlabel("Date")
    ax_aov.set_ylabel("Average Order Value (₹)")
    ax_aov.set_title("Average Order Value Over Time")
    ax_aov.grid(True)
    ax_aov.legend()
    fig_aov.autofmt_xdate(rotation=45)
    st.pyplot(fig_aov)

    st.markdown("---")


    # ===============================
    # Delivery Time Distribution
    # ===============================
    st.subheader("⏱️ Delivery Time Distribution") 
    fig4, ax4 = plt.subplots() 
    ax4.hist(filtered_df['Delivery_Time_Min'], bins=20, color='skyblue') 
    ax4.set_xlabel("Delivery Time (minutes)") 
    ax4.set_ylabel("Number of Orders")
    st.pyplot(fig4)

    # ===============================
    # Data Preview
    # ===============================
    with st.expander("🔍 View Dataset"):
        st.dataframe(filtered_df.head(100))

else:
    st.info("No data available for the selected date range.")