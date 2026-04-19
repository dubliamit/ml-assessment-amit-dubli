# Scenario: Promotion Effectiveness at a Fashion Retail Chain

A fashion retailer operates 50 stores across urban, semi-urban, and rural locations. Each month, the marketing team runs one of five promotions: Flat Discount, BOGO (Buy-One-Get-One), Free Gift with Purchase, Category-Specific Offer, and Loyalty Points Bonus. Stores vary in size, monthly footfall, local competition density, and customer demographics. The company wants to determine which promotion should be deployed in each store each month to maximise the number of items sold.

# B1. Problem Formulation

(a) - Formulate this as a machine learning problem. State clearly: what is the target variable, what are the candidate input features, and what type of ML problem is this? Justify your choice of problem type.

(Ans) - We can consider this as a Supervised machine learning Regression problem.
        Target variable - items_sold (monthly sales volume per store)
        Independent features
        1. Store id
        2. Store size
        3. Location type (Urban / Semi Urban / Rural)
        4. Promotion type
        5. month
        6. seasonality indicators
        7. festival/holidays
        8. footfall
        9. customer demographic
        10. Local competition density
Justification - We need to maximize the sales by predicting the expected outcome under different promotions.
