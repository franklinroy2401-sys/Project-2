# %%
import pandas as pd
import numpy as np
from IPython.display import display, Markdown

print("Libraries loaded successfully")

# %%
df = pd.read_csv("/Users/DELL/OneDrive/Desktop/Online Food deliver Analysis/Raw_Data.csv")
df

# %%
df.shape

# %%
df['Order_Date'] = pd.to_datetime(
    df['Order_Date'],
    format='%m/%d/%Y',
    errors='coerce'
)
df

df['Order_Date'].dt.year.value_counts()


# %%
df.shape

# %%
df.describe()

# %%
df['Order_Date_Missing'] = df['Order_Date'].isna().astype(int)
df

# %%
df.shape


# %%
df.shape


# %%
df.isnull().sum()


# %%
df['Order_Date'].min(), df['Order_Date'].max()


# %%
# =====================================
# Numerical Columns
# =====================================

# Customer_Age 
df['Customer_Age'].fillna(df['Customer_Age'].median(), inplace=True)
df['Customer_Age']
# Delivery_Time_Min
df['Delivery_Time_Min'].fillna(df['Delivery_Time_Min'].median(), inplace=True)
df['Delivery_Time_Min']

 # Distance_km
df['Distance_km'].fillna(df['Distance_km'].median(), inplace=True)
df['Distance_km']

# Order_Value
df['Order_Value'].fillna(df['Order_Value'].median(), inplace=True)
df['Order_Value']

# Delivery_Rating
df['Delivery_Rating'].fillna(df['Delivery_Rating'].median(), inplace=True)
df['Delivery_Rating']

# Discount Column Handling
df['Discount_Applied'] = pd.to_numeric(df['Discount_Applied'], errors='coerce')
df['Discount_Applied'].fillna(0, inplace=True)

# FINANCIAL LOGIC COLUMN : Final_Amount
df['Final_Amount'] = df['Final_Amount'].fillna(
    df['Order_Value'] - df['Discount_Applied']
)
df



# %%
df.shape

# %%
df[['Discount_Applied', 'Final_Amount']].isnull().sum()


# %%
df.isnull().sum()


# %%
# =====================================
# Categorical Columns
# =====================================

df['Customer_Gender'].fillna(df['Customer_Gender'].mode()[0], inplace=True)
df['Cuisine_Type'].fillna(df['Cuisine_Type'].mode()[0], inplace=True)

df['City'].fillna('Unknown', inplace=True)
df['Area'].fillna('Unknown', inplace=True)
df['Payment_Mode'].fillna('Unknown', inplace=True)
df

# %%
# =====================================
# Cancellation Reason (Business Logic)
# =====================================

df['Cancellation_Reason'].fillna('Not Cancelled', inplace=True)
df

# %%
# =====================================
# Peak Hour Column
# =====================================

df['Peak_Hour'].fillna('Non-Peak', inplace=True)
df

# %%
df.isnull().sum()

# %%
# Quick overview
df.info()
df.describe()  # For numeric columns
df.describe(include='object')  # For categorical columns


# %%
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')

# %%
sns.histplot(df['Order_Value'], bins=30, kde=True)
plt.title('Distribution of Order Value')
plt.show()

sns.boxplot(x=df['Order_Value'])
plt.title('Order Value Outliers')
plt.show()

# %%
# =========================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =========================================

# =================================
# Customer & Order Analysis 
# =================================
# 1. Identify Top-Spending Customers

top_customers = (
    df.groupby('Customer_ID')['Order_Value']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(top_customers)

# Plot
ax = top_customers.plot(kind='bar', figsize=(10,6))
plt.title('Top 10 Spending Customers')
plt.xlabel('Customer ID')
plt.ylabel('Total Order Value')

# Add values on top of bars
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2, p.get_height() + 5,  # adjust 5 if needed
            f'{p.get_height():.2f}', ha='center', va='bottom')

plt.show()

display(Markdown("""
## Business Insights  

**Identify top-spending customers**  
The results show that revenue is concentrated among a limited number of customers.  
The top 10 customers alone contribute a noticeably higher total order value compared to the rest of the customer base.  
This indicates a strong dependence on repeat high-value customers, suggesting that customer retention plays a more critical role in revenue growth than customer acquisition.  
"""))

# %%
# 2. Analyze age group vs order value 
df['Age_Group'] = pd.cut(
    df['Customer_Age'],
    bins=[18, 25, 35, 45, 60, 100],
    labels=['18-25', '26-35', '36-45', '46-60', '60+']
)

# Aggregate mean order value per age group (ignore empty categories)
age_group_order = df.groupby('Age_Group', observed=True)['Order_Value'].mean().reset_index()
age_group_order = age_group_order.dropna()  # remove NaN categories

print(age_group_order)

# Plot bar chart
plt.figure(figsize=(8,5))
sns.barplot(data=age_group_order, x='Age_Group', y='Order_Value', color='skyblue')

# Annotate values
for i, val in enumerate(age_group_order['Order_Value']):
    plt.text(i, val + 0.5, f"{val:.2f}", ha='center')

plt.title('Average Order Value by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Order Value')
plt.show()

display(Markdown("""
## Business Insights 
                 
**Analyze age group vs order value**  
Average order value increases across middle age groups, while younger age groups show comparatively lower spending per order.  
Spending behavior varies by age, with mid-age customers showing higher purchasing power.  
This highlights the need for age-based segmentation when designing pricing, promotions, and marketing campaigns.  
"""))



# %%
#3. Weekend vs weekday order patterns 

# Fix date parsing (IMPORTANT)
df['Order_Date'] = pd.to_datetime(
    df['Order_Date'],
    format='mixed',
    dayfirst=True,
    errors='coerce'
)

# Create Day_Type column
df['Day_Type'] = df['Order_Date'].dt.dayofweek.apply(
    lambda x: 'Weekend' if x >= 5 else 'Weekday'
)

# Count orders
day_type_orders = df['Day_Type'].value_counts()
print(day_type_orders)

# Plot
ax = day_type_orders.plot(kind='bar', color='skyblue')
plt.title('Weekend vs Weekday Order Patterns')
plt.xlabel('Day Type')
plt.ylabel('Number of Orders')

for i, val in enumerate(day_type_orders):
    ax.text(i, val + max(day_type_orders)*0.01, str(val), ha='center')

plt.show()

# Business Insight output
display(Markdown("""
## Business Insights  

**Weekend vs Weekday Order Patterns**  
Order volumes are noticeably higher on weekends compared to weekdays, indicating stronger demand during non-working days.  
This reflects lifestyle-driven ordering behavior, where customers prefer convenience during leisure time.  
Businesses should ensure higher delivery capacity on weekends while using targeted promotions to boost weekday demand.
"""))


# %%
# ==============================
# Revenue & Profit Analysis 
# ==============================
# 4. Monthly revenue trends

df['Order_Date'] = pd.to_datetime(
    df['Order_Date'],
    format='mixed',
    dayfirst=True,
    errors='coerce'
)

# Extract month
df['Month'] = df['Order_Date'].dt.to_period('M')

# Calculate monthly revenue
monthly_revenue = df.groupby('Month')['Order_Value'].sum()

# Convert to millions
monthly_revenue_m = monthly_revenue / 1_000_000

# Show data in output
print("Monthly Revenue (in Millions):")
print(monthly_revenue_m)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(monthly_revenue_m.index.astype(str), monthly_revenue_m, marker='o')

# Add value labels on points
for x, y in zip(monthly_revenue_m.index.astype(str), monthly_revenue_m):
    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')

plt.title('Monthly Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Revenue (in Millions)')
plt.show()

display(Markdown("""
## Business Insights 

**Monthly Revenue Trends**
Monthly revenue shows noticeable variation across different months rather than remaining constant. This indicates the presence of seasonality and demand fluctuations over time.
Revenue planning should account for peak and low-demand months. Operational capacity and marketing efforts can be aligned with high-revenue months, while promotions can be used to support weaker periods     
"""))                                                                       

# %%
# 5. Impact of Discounts on Profit 
# Step 1: Convert Discount_Applied to numeric (if it's in string/percentage format)
if df['Discount_Applied'].dtype == 'object':
    # Remove '%' if present and convert to float
    df['Discount_numeric'] = df['Discount_Applied'].astype(str).str.rstrip('%').astype(float)
else:
    df['Discount_numeric'] = df['Discount_Applied'].astype(float)

# Step 2: Group by discount and calculate average profit
discount_profit = df.groupby('Discount_numeric')['Profit_Margin'].mean().reset_index()

# Step 2a: Show data in output
print("Average Profit by Discount (%):")
print(discount_profit)

# Step 3: Plot
plt.figure(figsize=(10, 5))
plt.plot(discount_profit['Discount_numeric'], discount_profit['Profit_Margin'], marker='o', color='teal')

# Add value labels on each point
for x, y in zip(discount_profit['Discount_numeric'], discount_profit['Profit_Margin']):
    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=9)

plt.title('Impact of Discounts on Profit', fontsize=14)
plt.xlabel('Discount (%)', fontsize=12)
plt.ylabel('Average Profit', fontsize=12)
plt.grid(True)
plt.show()

display(Markdown("""
## Business Insights 
                 
**Impact of discounts on profit**

The analysis shows that average profit margin decreases as discount percentage increases.
Higher discounts lead to reduced profitability despite potentially higher order volumes.  
Discounts should be applied selectively rather than uniformly. 
Controlled discounting strategies help balance customer acquisition with long-term profitability.
"""))  
                                             


# %%
# 6. High-revenue cities and cuisines 
# Column names in your dataset
# -----------------------------
# ASSUMED COLUMN NAMES
# -----------------------------
# City column        -> 'City'
# Cuisine column     -> 'Cuisine_Type'
# Revenue column     -> 'Order_Value'

# =============================
# TOP 10 HIGH-REVENUE CITIES
# =============================

city_revenue = (
    df.groupby('City')['Order_Value']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

# Show data in output
print("Top 10 High-Revenue Cities:")
print(city_revenue)

# Convert revenue to millions
city_revenue['Revenue_M'] = city_revenue['Order_Value'] / 1_000_000

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=city_revenue,
    x='City',
    y='Revenue_M',
    hue='City',
    legend=False
)

# Add values on bars
for i, value in enumerate(city_revenue['Revenue_M']):
    plt.text(i, value, f"{value:.2f}M", ha='center', va='bottom')

plt.title('Top 10 High-Revenue Cities (in Millions)')
plt.xlabel('City')
plt.ylabel('Total Revenue (Millions)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =============================
# TOP 10 HIGH-REVENUE CUISINES
# =============================

cuisine_revenue = (
    df.groupby('Cuisine_Type')['Order_Value']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

# Show data in output
print("\nTop 10 High-Revenue Cuisines:")
print(cuisine_revenue)

# Convert revenue to millions
cuisine_revenue['Revenue_M'] = cuisine_revenue['Order_Value'] / 1_000_000

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=cuisine_revenue,
    x='Cuisine_Type',
    y='Revenue_M',
    hue='Cuisine_Type',
    legend=False
)

# Add values on bars
for i, value in enumerate(cuisine_revenue['Revenue_M']):
    plt.text(i, value, f"{value:.2f}M", ha='center', va='bottom')

plt.title('Top 10 High-Revenue Cuisines (in Millions)')
plt.xlabel('Cuisine')
plt.ylabel('Total Revenue (Millions)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
display(Markdown("""
### Business Insights  
**Revenue & Profit Analysis**  

#### High-Revenue Cities and Cuisines  

**High-Revenue Cities**  
The analysis shows that total revenue is highly concentrated in a limited number of cities.  
The top 10 cities contribute a significantly higher share of overall revenue compared to other locations.  
This indicates strong regional demand and suggests that expanding operations, delivery capacity, and localized marketing efforts in these cities can drive sustained revenue growth.

**High-Revenue Cuisines**  
Revenue is dominated by a small set of cuisines, reflecting clear customer preferences.  
The top cuisines generate a disproportionately large portion of total order value, while others contribute marginally.  
This highlights the importance of prioritizing popular cuisines for menu expansion, promotions, and restaurant partnerships to maximize revenue potential.
"""))


# %%
# ========================
# Delivery Performance 
# ========================
# 7. Average delivery time by city

# Column names
city_col = 'City'
delivery_time_col = 'Delivery_Time_Min'

# Step 1: Calculate average delivery time per city
avg_delivery = df.groupby(city_col, observed=True)[delivery_time_col].mean().sort_values().reset_index()

# Step 2: Show the output table
print("Average Delivery Time by City:")
print(avg_delivery)

# Step 3: Plot
plt.figure(figsize=(12,6))
sns.barplot(data=avg_delivery, x=city_col, y=delivery_time_col, color='skyblue')  # use color instead of palette to remove warning

# Add values on bars
for i, v in enumerate(avg_delivery[delivery_time_col]):
    plt.text(i, v + 0.5, f'{v:.1f} min', ha='center', va='bottom', fontsize=9)

plt.title('Average Delivery Time by City', fontsize=14)
plt.xlabel('City', fontsize=12)
plt.ylabel('Average Delivery Time (Minutes)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

display(Markdown("""
**Business Insight:** 
### Average Delivery Time by City 
Average delivery time varies significantly across cities.  
Some cities experience consistently higher delivery times, indicating potential operational or traffic-related inefficiencies.
"""))


# %%
# 8.  Distance vs delivery delay analysis

# Columns
distance_col = 'Distance_km'
delivery_time_col = 'Delivery_Time_Min'

# Remove rows with missing values
df_clean = df.dropna(subset=[distance_col, delivery_time_col])

# Step 1: Create larger distance bins (5 km)
bins = np.arange(0, df_clean[distance_col].max() + 5, 5)
df_clean['Distance_bin'] = pd.cut(df_clean[distance_col], bins)

# Step 2: Calculate average delivery time per distance bin
distance_delay = df_clean.groupby('Distance_bin', observed=True)[delivery_time_col].mean().reset_index()

# Step 3: Format distance bin labels nicely
distance_delay['Distance_bin_str'] = distance_delay['Distance_bin'].apply(lambda x: f"{int(x.left)}-{int(x.right)} km")

# Step 4: Show output table
print("Average Delivery Time by Distance Bin:")
print(distance_delay[['Distance_bin_str', delivery_time_col]])

# Step 5: Plot
plt.figure(figsize=(12,6))
sns.barplot(data=distance_delay, x='Distance_bin_str', y=delivery_time_col, color='skyblue')

# Add values on bars
for i, v in enumerate(distance_delay[delivery_time_col]):
    plt.text(i, v + 1.5, f'{v:.1f} min', ha='center', va='bottom', fontsize=8)

plt.title('Average Delivery Time vs Distance', fontsize=14)
plt.xlabel('Distance Bin', fontsize=12)
plt.ylabel('Average Delivery Time (Minutes)', fontsize=12)
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()
display(Markdown("""
**Business Insight:**  
### Distance vs Delivery Delay Analysis
Delivery time generally increases as distance increases.  
Delays observed even at shorter distances suggest that factors beyond distance impact delivery efficiency.
"""))


# %%
# 9. Delivery rating vs delivery time 

delivery_time_col = 'Delivery_Time_Min'
rating_col = 'Delivery_Rating'

# Remove missing values
df_clean = df.dropna(subset=[delivery_time_col, rating_col]).copy()

# Create 30-minute bins
bins = np.arange(0, df_clean[delivery_time_col].max() + 30, 30)
df_clean['Delivery_Time_Bin'] = pd.cut(df_clean[delivery_time_col], bins)

# Calculate average rating per bin
rating_vs_time = (
    df_clean
    .groupby('Delivery_Time_Bin', observed=True)[rating_col]
    .mean()
    .reset_index()
)

# Create readable labels
rating_vs_time['Delivery_Time_Range'] = rating_vs_time['Delivery_Time_Bin'].apply(
    lambda x: f"{int(x.left)}–{int(x.right)} min"
)

# Round for clean output
rating_vs_time[rating_col] = rating_vs_time[rating_col].round(2)

# ===== OUTPUT TABLE =====
print("Average Delivery Rating by Delivery Time:")
print(rating_vs_time[['Delivery_Time_Range', rating_col]])

# ===== VISUALIZATION =====
plt.figure(figsize=(10,5))
sns.barplot(
    data=rating_vs_time,
    x='Delivery_Time_Range',
    y=rating_col
)

for i, v in enumerate(rating_vs_time[rating_col]):
    plt.text(i, v + 0.03, f'{v}', ha='center', fontsize=9)

plt.title('Delivery Rating vs Delivery Time')
plt.xlabel('Delivery Time (Minutes)')
plt.ylabel('Average Delivery Rating')
plt.ylim(2.7, 3.2)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
display(Markdown("""
**Business Insight:**  
### Delivery Rating vs Delivery Time
Customer ratings decrease as delivery time increases.  
Faster deliveries are strongly associated with higher customer satisfaction.
"""))



# %%
# ==================================
# Restaurant Performance 
# ==================================

# 10. Top-rated restaurants 

restaurant_col = 'Restaurant_Name'
rating_col = 'Delivery_Rating'

# Remove missing values
df_clean = df.dropna(subset=[restaurant_col, rating_col]).copy()

# Average rating per restaurant
top_rated = df_clean.groupby(restaurant_col)[rating_col].mean().reset_index()

# Sort and take top 10
top_rated = top_rated.sort_values(rating_col, ascending=False).head(10)
top_rated[rating_col] = top_rated[rating_col].round(2)

# ===== OUTPUT =====
print("Top 10 Rated Restaurants:")
print(top_rated)

# ===== PLOT =====
plt.figure(figsize=(10,5))
sns.barplot(data=top_rated, x=rating_col, y=restaurant_col)

for i, v in enumerate(top_rated[rating_col]):
    plt.text(v + 0.01, i, f'{v}', va='center')

plt.title('Top 10 Rated Restaurants')
plt.xlabel('Average Rating')
plt.ylabel('Restaurant')
plt.xlim(0,5)
plt.tight_layout()
plt.show()

display(Markdown("""
**Business Insight:**
### Top-Rated Restaurants  
A small group of restaurants consistently receives higher average ratings.  
These restaurants demonstrate better service quality and reliability, making them strong candidates for promotions and featured listings.
"""))


# %%
# 11. Cancellation Rate by Restaurant (Vertical Bar Chart)

restaurant_col = 'Restaurant_Name'
status_col = 'Order_Status'

# Step 1: Create cancellation flag
df['Is_Cancelled'] = df[status_col].apply(lambda x: 1 if x == 'Cancelled' else 0)

# Step 2: Calculate cancellation rate per restaurant
cancel_rate = (
    df.groupby(restaurant_col)['Is_Cancelled']
      .mean()
      .reset_index()
)

# Convert to percentage
cancel_rate['Cancellation_Rate_%'] = (cancel_rate['Is_Cancelled'] * 100).round(2)

# Step 3: Select top 10 restaurants
top_cancel = cancel_rate.sort_values(
    by='Cancellation_Rate_%',
    ascending=False
).head(10)

# Step 4: Show output
print("Top 10 Restaurants by Cancellation Rate:")
print(top_cancel[[restaurant_col, 'Cancellation_Rate_%']])

# Step 5: Plot (Vertical Bar Chart)
plt.figure(figsize=(10,6))
sns.barplot(
    data=top_cancel,
    x=restaurant_col,
    y='Cancellation_Rate_%'
)

# Add values on bars
for i, v in enumerate(top_cancel['Cancellation_Rate_%']):
    plt.text(i, v + 0.3, f'{v}%', ha='center', fontsize=9)

plt.title('Top 10 Restaurants by Cancellation Rate', fontsize=14)
plt.xlabel('Restaurant')
plt.ylabel('Cancellation Rate (%)')
plt.xticks(rotation=45)

# Improve outer box visibility
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.show()

display(Markdown("""
**Business Insight:** 
### Cancellation Rate by Restaurant 
A few restaurants show significantly higher cancellation rates compared to others.  
High cancellation rates indicate operational issues such as order readiness or stock availability, which negatively impact customer experience.
"""))


# %%
# 12. Cuisine-wise Performance 

cuisine_col = 'Cuisine_Type'
rating_col = 'Delivery_Rating'

df_clean = df.dropna(subset=[cuisine_col, rating_col]).copy()

# Average rating by cuisine
cuisine_perf = df_clean.groupby(cuisine_col)[rating_col].mean().reset_index()
cuisine_perf[rating_col] = cuisine_perf[rating_col].round(2)

# Sort by rating
cuisine_perf = cuisine_perf.sort_values(rating_col, ascending=False)

# ===== OUTPUT =====
print("Cuisine-wise Performance:")
print(cuisine_perf)

# ===== PLOT =====
plt.figure(figsize=(10,5))
sns.barplot(data=cuisine_perf, x=rating_col, y=cuisine_col)

for i, v in enumerate(cuisine_perf[rating_col]):
    plt.text(v + 0.01, i, f'{v}', va='center')

plt.title('Cuisine-wise Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Cuisine')
plt.xlim(0,5)
plt.tight_layout()
plt.show()
display(Markdown("""
**Business Insight:** 
### Cuisine-Wise Performance 
Certain cuisines consistently achieve higher average ratings than others.  
This reflects stronger customer preference and satisfaction for specific cuisines, suggesting opportunities to promote high-performing cuisine categories.
"""))



# %%
# =====================================
# Operational Insights
# =====================================
# 13. Demand Level Analysis (Operational Peak Load)
# =====================================

# Step 1: Calculate daily order count
daily_orders = df.groupby('Order_Date').size()

# Step 2: Map daily order count to dataframe
df['Daily_Order_Count'] = df['Order_Date'].map(daily_orders)

# Step 3: Create Demand Level feature
average_daily_orders = daily_orders.mean()

df['Demand_Level'] = df['Daily_Order_Count'].apply(
    lambda x: 'High Demand' if x > average_daily_orders else 'Low Demand'
)

# Step 4: Aggregate demand level data
demand_analysis = df['Demand_Level'].value_counts().reset_index()
demand_analysis.columns = ['Demand_Level', 'Total_Orders']

print("Demand Level Order Distribution:")
print(demand_analysis)

# Step 5: Visualization
plt.figure(figsize=(6,4))
sns.barplot(data=demand_analysis, x='Demand_Level', y='Total_Orders')

# Add value labels
for i, v in enumerate(demand_analysis['Total_Orders']):
    plt.text(i, v, v, ha='center', va='bottom')

plt.title('Demand Level Analysis (Operational Peak Load)')
plt.xlabel('Demand Level')
plt.ylabel('Total Orders')

# Improve axis visibility
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.show()

# Step 6: Business Insight
display(Markdown("""
**Business Insight:**
### Demand Level Analysis (Operational Peak Load)

Orders are unevenly distributed across days, with high-demand days contributing a larger share 
of total orders. These high-demand periods represent operational peak load conditions, 
requiring increased delivery partner availability and efficient logistics planning 
to avoid delivery delays and service degradation.
"""))


# %%
# 14. Payment Mode Preferences 

payment_col = 'Payment_Mode'

# Count orders per payment mode
payment_pref = df[payment_col].value_counts().reset_index()
payment_pref.columns = ['Payment_Mode', 'Total_Orders']

# Calculate total orders
total_orders = payment_pref['Total_Orders'].sum()

# Show output
print("Payment Mode Preferences:")
print(payment_pref)
print(f"\nTotal Orders: {total_orders}")

# Function to show counts instead of percentages
def show_counts(pct):
    count = int(round(pct * total_orders / 100))
    return f'{count}'

# Colors
colors = ['#66b3ff', '#99ff99', '#ffcc99', '#ff9999', '#c2c2f0']

# Plot pie chart
plt.figure(figsize=(7,7))
plt.pie(
    payment_pref['Total_Orders'],
    labels=payment_pref['Payment_Mode'],
    autopct=show_counts,
    startangle=140,
    colors=colors
)

plt.title(f'Payment Mode Preferences\nTotal Orders: {total_orders}')
plt.tight_layout()
plt.show()

display(Markdown("""
**Business Insight:** 
### Payment Mode Preferences 
Digital payment methods account for the majority of transactions compared to cash payments.  
This indicates a strong customer preference for cashless transactions and supports continued focus on digital payment integrations.
"""))



# %%
# 15. Cancellation Reason Analysis
# Count cancellations by reason
cancel_reason = df['Cancellation_Reason'].value_counts()
percentages = (cancel_reason / cancel_reason.sum() * 100).round(2)

# Display table
print("Cancellation Reasons:\n")
for reason, count in cancel_reason.items():
    print(f"{reason}: {count} ({percentages[reason]}%)")

# Plot vertical bar chart
plt.figure(figsize=(12,6))
bars = plt.bar(cancel_reason.index, cancel_reason.values, color=plt.cm.viridis(range(len(cancel_reason))))

# Add counts and percentages on top
for bar, perc in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'{int(bar.get_height())} ({perc}%)', ha='center', va='bottom', fontsize=10)

# Title without total cancellations
plt.title('Cancellation Reason Analysis', fontsize=14)
plt.xlabel('Cancellation Reason')
plt.ylabel('Total Cancellations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

display(Markdown("""
**Business Insight:** 
### Cancellation Reason Analysis 
Most cancellations are driven by delivery delays and restaurant-related issues rather than customer actions.  
Addressing operational inefficiencies can significantly reduce cancellation rates and improve customer satisfaction.
"""))



# %%
# =============================
# Feature Engineering 
# =============================
# 1. Order day type (Weekday / Weekend)
df['Order_Day_Type'] = np.where(
    df['Order_Date'].notna(),
    np.where(df['Order_Date'].dt.dayofweek >= 5, 'Weekend', 'Weekday'),
    'Unknown'
)
df

print("Business Insights")
print("Most orders are placed on weekdays, indicating higher customer activity during the workweek."  
      "Weekend promotions or campaigns can be used to boost sales during lower-order days.")
print("Order Day Type Distribution:")

print(df['Order_Day_Type'].value_counts())

# Percentage
print("\nPercentage Distribution:")
print(df['Order_Day_Type'].value_counts(normalize=True) * 100)


# %%
df['Order_Time'].head()
df['Order_Time'].dtype


# %%
# 2. Peak hour indicator
def pseudo_peak(delivery_time):
    if delivery_time >= 60:
        return 'High_Demand_Period'
    else:
        return 'Normal_Demand_Period'

df['Demand_Period_Proxy'] = df['Delivery_Time_Min'].apply(pseudo_peak)
df

print("Business Insight:")
print("I created proxies to identify high-demand and peak delivery periods. "
      "Demand_Period_Proxy highlights periods with longer delivery times, indicating high demand, "
      "while Peak_Hour_Indicator flags peak hours operationally, including missing data as 'Unknown'. "
      "These insights help in resource planning and optimizing delivery schedules.")

def peak_hour_proxy(delivery_time):
    if pd.isna(delivery_time):
        return 'Unknown'
    elif delivery_time >= 60:
        return 'Peak'
    else:
        return 'Non-Peak'

df['Peak_Hour_Indicator'] = df['Delivery_Time_Min'].apply(peak_hour_proxy)
df

# Demand_Period_Proxy
demand_counts = df['Demand_Period_Proxy'].value_counts(normalize=True) * 100
print("Demand Period Percentage Distribution:")
print(demand_counts)

# Peak_Hour_Indicator
peak_counts = df['Peak_Hour_Indicator'].value_counts(normalize=True) * 100
print("\nPeak Hour Percentage Distribution:")
print(peak_counts)


# %%
# 3. Profit margin percentage
df['Profit_Margin_Percentage'] = np.where(
    df['Final_Amount'] > 0,
    ((df['Final_Amount'] - df['Order_Value']) / df['Final_Amount']) * 100,
    np.nan
)

df['Profit_Margin_Percentage'] = ((df['Final_Amount'] - df['Order_Value']) / df['Final_Amount']) * 100
df['Profit_Margin_Percentage'] = df['Profit_Margin_Percentage'].round(2)

print("Business Insight:")
print("High Demand Period orders can still have negative profit margins if costs are high" 
      "Normal Demand Period orders also show negative margins for loss-making orders." 
      "This highlights pricing or cost issues in different demand periods.")

# Count percentage of each category
distribution = df['Profit_Margin_Percentage'].value_counts(normalize=True) * 100
distribution = distribution.sort_index()  # optional: sort by the bin order

print("Profit Margin Percentage Distribution (%):")
print(distribution)



# %%
import numpy as np

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Optional: replace NaN with 0 if you want to avoid nulls
df['Profit_Margin_Percentage'].fillna(0, inplace=True)


# %%
# 4. Delivery performance categories 
def delivery_performance(time):
    if time <= 30:
        return 'Fast'
    elif time <= 60:
        return 'Average'
    else:
        return 'Slow'

df['Delivery_Performance_Category'] = df['Delivery_Time_Min'].apply(delivery_performance)
df

print("Business Insights")
print ("Most orders are delivered within the average timeframe, indicating generally stable logistics,"  
       "While some deliveries are fast, a small percentage are slow, highlighting opportunities to optimize operations,"  
       "improve customer satisfaction, and reduce delays in critical areas.")


# Percentage distribution
delivery_dist = df['Delivery_Performance_Category'].value_counts(normalize=True) * 100
delivery_dist = delivery_dist.round(2)  # round to 2 decimals

print("Delivery Performance Percentage Distribution (%):")
print(delivery_dist)




# %%
# 5. Customer age groups 
def age_group(age):
    if age < 25:
        return 'Young'
    elif age <= 40:
        return 'Adult'
    else:
        return 'Senior'

df['Customer_Age_Group'] = df['Customer_Age'].apply(age_group)
df

print("Business Insights")
print("The majority of customers fall within the Young Adult and Adult categories, suggesting these age groups drive most orders."  
      "Marketing and engagement strategies can be tailored to target these key segments for higher impact")

# Calculate percentage distribution
age_dist = df['Customer_Age_Group'].value_counts(normalize=True) * 100
age_dist = age_dist.round(2)

print("Customer Age Group Percentage Distribution (%):")
print(age_dist)


# %%
import pandas as pd
from sqlalchemy import create_engine, text

# ==================================================
# Step 1: (Run ONCE) Create database if not exists
# ==================================================
engine_server = create_engine(
    "mysql+pymysql://root:RoyFrank@localhost:3306/"
)

with engine_server.connect() as conn:
    conn.execute(text("CREATE DATABASE IF NOT EXISTS food_delivery_db;"))
    print("Database 'food_delivery_db' is ready.")

# ==========================================
# Step 2: Connect to the specific database
# ==========================================
engine_db = create_engine(
    "mysql+pymysql://root:RoyFrank@localhost:3306/food_delivery_db"
)

# ===========================================
# Step 3: Upload DataFrame to MySQL
# ===========================================
table_name = "food_orders_analysis"

# Make sure 'df' is already defined with your processed data
df.to_sql(
    name=table_name,
    con=engine_db,
    if_exists="replace",
    index=False
)

print(f"DataFrame uploaded to MySQL table '{table_name}' successfully!")

# ==============================================
# Step 4: Verify data insertion (first 10 rows)
# ==============================================
with engine_db.connect() as conn:
    result = conn.execute(
        text("SELECT * FROM food_orders_analysis LIMIT 10;")
    )

    print("\nFirst 10 rows of the table:")
    for row in result:
        print(row)

# ==============================================
# Step 5: Export the SQL table back to CSV
# ==============================================
query = f"SELECT * FROM {table_name};"
df_from_sql = pd.read_sql(query, engine_db)

# Export to CSV
csv_file = "processed_food_orders.csv"
df_from_sql.to_csv(csv_file, index=False)
print(f"\nSQL table '{table_name}' exported to CSV file '{csv_file}' successfully!")




