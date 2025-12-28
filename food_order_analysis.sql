USE food_delivery_db;
SHOW TABLES;
SELECT * FROM food_orders_analysis LIMIT 5;

-- ============================
-- Customer & Order Analysis 
-- ============================

-- 1. Identify top-spending customers 
SELECT 
    Customer_ID,
    SUM(Final_Amount) AS Total_Spent,
    COUNT(Order_ID) AS Total_Orders
FROM food_orders_analysis
GROUP BY Customer_ID
ORDER BY Total_Spent DESC
LIMIT 10;

-- 2. Analyze age group vs order value 
SELECT 
    Customer_Age_Group,
    AVG(Final_Amount) AS Avg_Order_Value,
    COUNT(*) AS Total_Orders
FROM food_orders_analysis
GROUP BY Customer_Age_Group
ORDER BY Avg_Order_Value DESC;  

-- 3. Weekend vs weekday order patterns  
SELECT 
    Order_Day_Type,
    COUNT(*) AS Total_Orders,
    AVG(Final_Amount) AS Avg_Order_Value
FROM food_orders_analysis
GROUP BY Order_Day_Type;

-- =============================
-- Revenue & Profit Analysis
-- ============================= 

-- 4. Monthly revenue trends  
SELECT 
    DATE_FORMAT(Order_Date, '%Y-%m') AS Month,
    SUM(Final_Amount) AS Monthly_Revenue
FROM food_orders_analysis
GROUP BY DATE_FORMAT(Order_Date, '%Y-%m')
ORDER BY Month;

-- 5. Impact of discounts on profit 
SELECT 
    CASE 
        WHEN Order_Value > Final_Amount THEN 'Discount Applied'
        ELSE 'No Discount'
    END AS Discount_Category,
    ROUND(AVG(Profit_Margin_Percentage), 2) AS Avg_Profit_Margin
FROM food_orders_analysis
GROUP BY 
    CASE 
        WHEN Order_Value > Final_Amount THEN 'Discount Applied'
        ELSE 'No Discount'
    END;

-- 6.  High-revenue cities and cuisines 
-- High-revenue cities
SELECT 
    City,
    SUM(Final_Amount) AS Total_Revenue
FROM food_orders_analysis
GROUP BY City
ORDER BY Total_Revenue DESC
LIMIT 10; 

-- High-revenue cuisines 
SELECT 
    Cuisine_Type,
    SUM(Final_Amount) AS Total_Revenue
FROM food_orders_analysis
GROUP BY Cuisine_Type
ORDER BY Total_Revenue DESC
LIMIT 10;

-- Delivery Performance 
-- 7. Average delivery time by city  
SELECT 
    City,
    AVG(Delivery_Time_Min) AS Avg_Delivery_Time
FROM food_orders_analysis
GROUP BY City
ORDER BY Avg_Delivery_Time; 

-- 8. Distance vs delivery delay analysis 
SELECT 
    CASE
        WHEN Distance_km < 5 THEN 'Short Distance'
        WHEN Distance_km BETWEEN 5 AND 10 THEN 'Medium Distance'
        ELSE 'Long Distance'
    END AS Distance_Category,
    ROUND(AVG(Delivery_Time_Min), 2) AS Avg_Delivery_Time_Min
FROM food_orders_analysis
GROUP BY Distance_Category
ORDER BY Avg_Delivery_Time_Min;  

-- 9. Delivery rating vs delivery time 
SELECT 
    Delivery_Rating,
    AVG(Delivery_Time_Min) AS Avg_Delivery_Time
FROM food_orders_analysis
GROUP BY Delivery_Rating
ORDER BY Delivery_Rating DESC; 

-- Restaurant Performance  
-- 10. Top-rated restaurants  
SELECT 
    Restaurant_Name,
    AVG(Delivery_Rating) AS Avg_Rating,
    COUNT(*) AS Total_Orders
FROM food_orders_analysis
GROUP BY Restaurant_Name
HAVING COUNT(*) > 20
ORDER BY Avg_Rating DESC
LIMIT 10;

-- 11.  Cancellation rate by restaurant 
SELECT 
    Restaurant_Name,
    COUNT(CASE WHEN Order_Status = 'Cancelled' THEN 1 END) * 100.0 / COUNT(*) 
        AS Cancellation_Rate
FROM food_orders_analysis
GROUP BY Restaurant_Name
ORDER BY Cancellation_Rate DESC; 

-- 12. Cuisine-wise performance 
SELECT 
    cuisine_type AS Cuisine,
    COUNT(*) AS Total_Orders,
    AVG(Final_Amount) AS Avg_Order_Value
FROM food_orders_analysis
GROUP BY cuisine_type
ORDER BY Total_Orders DESC; 

-- Operational Insights 
-- 13. Peak hour demand analysis 
SELECT 
    Peak_Hour_Indicator,
    COUNT(*) AS Total_Orders
FROM food_orders_analysis
GROUP BY Peak_Hour_Indicator; 

-- 14. Payment mode preferences  
SELECT 
    Payment_Mode,
    COUNT(*) AS Total_Orders,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM food_orders_analysis), 2)
        AS Percentage_Share
FROM food_orders_analysis
GROUP BY Payment_Mode
ORDER BY Total_Orders DESC; 

-- 15. Cancellation reason analysis 
SELECT 
    Cancellation_Reason,
    COUNT(*) AS Total_Cancellations
FROM food_orders_analysis
WHERE Order_Status = 'Cancelled'
GROUP BY Cancellation_Reason
ORDER BY Total_Cancellations DESC;




















