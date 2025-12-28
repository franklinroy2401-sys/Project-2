🍔 Online Food Delivery Analytics Dashboard

An end-to-end Data Analytics project that demonstrates a complete real-world analyst workflow using Python, SQL (MySQL), and Streamlit.
The project focuses on extracting business insights from online food delivery data through EDA, feature engineering, SQL analysis, and interactive visualization.

📌 Project Objective

To analyze online food delivery data and uncover insights related to:

Customer ordering behavior

Revenue and profit trends

Delivery performance

Restaurant efficiency

Operational patterns

🛠️ Tech Stack

Programming & Analysis

Python

Pandas

NumPy

Visualization

Matplotlib

Streamlit

Database

MySQL

SQLAlchemy

MySQL Connecto

Tech Stack

Python

Pandas

MySQL

Streamlit

Project Workflow

1️⃣ Exploratory Data Analysis (EDA)

Data type validation

Missing value treatment

Distribution analysis

Outlier detection

Initial business observations

2️⃣ Feature Engineering

Created meaningful analytical features such as:

Order Day Type (Weekday / Weekend)

Peak Hour Indicator

Distance Category (Short / Medium / Long)

Profit Margin

Demand Period Proxy

The final cleaned dataset was saved as:
processed_food_orders.csv


3️⃣ SQL Analysis (15 Business Questions)

After feature engineering, the processed dataset was loaded into MySQL, and 15 business-oriented SQL queries were executed.

🧍 Customer & Order Analysis

 1. Identify top-spending customers

 2. Analyze age group vs order value

 3. Weekend vs weekday order patterns

💰 Revenue & Profit Analysis

 4. Monthly revenue trends

 5. Impact of discounts on profit

 6. High-revenue cities and cuisines

🚚 Delivery Performance

 7. Average delivery time by city
 
 8. Distance vs delivery delay analysis

 9. Delivery rating vs delivery time

🍽️ Restaurant Performance

 10. Top-rated restaurants

 11. Cancellation rate by restaurant

 12. Cuisine-wise performance

⚙️  Operational Insights

 13. Peak hour demand analysis

 14. Payment mode preferences

 15. Cancellation reason analysis
     
4️⃣ Streamlit Dashboard

An interactive Streamlit dashboard was built using the processed data.

📊 Key Performance Indicators (KPIs)

-Total Orders

-Total Revenue

-Average Order Value

-Average Delivery Time

-Cancellation Rate

-Average Delivery Rating

-Profit Margin %

📈 Visualizations

-Orders trend over time

-Revenue trend

-Average order value over time

-Profit margin trend

-Delivery time distribution

-Cancellation analysis

-Rating distribution

The dashboard includes date range filters for interactive exploration.

Step 1: Install dependencies

pip install -r requirements.txt

Step 2:  Run the Streamlit dashboard

streamlit run app.py

📈 Key Business Insights

⦁	Weekend orders generate higher revenue than weekdays

⦁	High discounts significantly reduce profit margins

⦁	Longer delivery times negatively impact customer ratings

⦁	Evening hours experience peak order demand

⦁	Certain cuisines consistently outperform others in revenue

🎯 Key Learnings

⦁	End-to-end data analytics project execution

⦁	Writing business-driven SQL queries

⦁	Feature engineering for analytics

⦁	Interactive dashboard development using Streamlit

⦁	Translating data into actionable insights

### Dataset

The dataset `Processed_food_order.csv` is large, so it has been compressed into `Processed_food_order.zip`. 

▶️ How to Run the Project (Local Setup)

1️⃣ Create MySQL Database

Create a MySQL database for the project:

CREATE DATABASE food_delivery_db;

2️⃣ Load Processed Data into MySQL

Run the Python script or notebook that uploads the processed dataset into MySQL
(Table used: food_orders_analysis).

This step loads the feature-engineered dataset into the database for SQL analysis.

3️⃣ Execute SQL Analysis Queries

Run the SQL file containing all 15 business analysis queries:

SOURCE food_delivery_analysis_queries.sql;

4️⃣ Install Python Dependencies

Install all required Python libraries:

pip install -r requirements.txt

5️⃣ Run the Streamlit Dashboard

Start the Streamlit application:

streamlit run app.py

6️⃣ Access the Dashboard

Open your browser and visit:

http://localhost:8501



