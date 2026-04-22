# Scenario: Promotion Effectiveness at a Fashion Retail Chain

A fashion retailer operates 50 stores across urban, semi-urban, and rural locations. Each month, the marketing team runs one of five promotions: Flat Discount, BOGO (Buy-One-Get-One), Free Gift with Purchase, Category-Specific Offer, and Loyalty Points Bonus. Stores vary in size, monthly footfall, local competition density, and customer demographics. The company wants to determine which promotion should be deployed in each store each month to maximise the number of items sold.

# B1. Problem Formulation

**(a)** - Formulate this as a machine learning problem. State clearly: what is the target variable, what are the candidate input features, and what type of ML problem is this? Justify your choice of problem type.

**(Ans)** - We can consider this as a Supervised machine learning Regression problem.
Target variable - items sold (monthly sales volume per store)
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
    
Justification - We need to maximize the sales by predicting the expected outcome under different promotions using regression model. And the best promotion can be selected by choosing the one with highest predicted items sold.

**(b)** - The company currently measures performance using total sales revenue. Explain why using items sold (sales volume) is a more reliable target variable for this problem. What broader principle does this illustrate about target variable selection in real-world ML projects?

**(Ans)** - When promotions are used, sale quantity of items can be more but on the other hand sales revenue may be less because of the promotions applied thus using items sold (sales volume) is a more reliable target variable for this problem. A promotion may sell more items but generate lower revenue.
Principally target should be such that truly represents what you want to predict, not something indirectly affected by other factors.
Proxy variables (like revenue) can be misleading because they mix multiple effects (price + demand). A good target focuses on actual behavior (like items sold), giving clearer learning.This helps the model make decisions based on real patterns, not distorted signals.

**(c)** - A junior analyst suggests running one single global model across all 50 stores. Propose and justify an alternative modelling strategy that accounts for the fact that stores in different locations respond very differently to the same promotion.

**(Ans)** - As the stores are at different location with different customer type or location type so responce to every promotion will be different as per the customer behavior, competition, income level etc and so running one single global model across all 50 stores will not be faesible. Instead we can build the seperate models for Urban, semi urban and Rutral stores OR we can also build models based on location with promotion type etc.

# B2. Data and EDA Strategy 
**(a)** - The raw data arrives in four separate tables: transactions, store attributes, promotion details, and a calendar (with weekend and festival flags). Describe how you would join these tables. What is the grain of the final modelling dataset (one row = what?), and what aggregations would you perform before modelling?

**(Ans)** - Transaction table (fact table) - 1. transaction_date, store_id and items_sold.
Dimension table 1 - store_id(PK) , store_size, location_type, demographic
Dimension table 2 - store_id/transaction_date (PK), promotion_type.
Dimension table 3 - transaction_date(PK), weekend_flag, holiday_flag (calender table)

Grain of the final modelling dataset - **one row = promotion type used in a one particular store for one month**  and decision is taken on the data taken for all store with all different promotions for one month.
aggregation can be done on 
1. items_sold per month per store (summation)
2. total monthly footfall (summation)
3. weekends and holidays in that month and its effect on sales
4. assigning the monthly promotions type per store.

**(b)** - Describe the EDA you would perform before building a model. Specify at least four analyses or charts, what you would look for in each, and how the findings would influence your feature engineering or modelling decisions.

**(Ans)** - 1. Skewness or outliers in items_sold will give us the distribution of target variable.
2. Time series plot (Line chart) to check trend over time mostly the seasonality or weekend effect on the sales.
3. Bar chart showing sales at different locations as a group (Urban, semi urban and Rural).
4. Corelation Heatmap helps in checking the relationship between numerical variables like footfall, store_size, local competition density, items_sold etc.
5. checking linear / non linear relationship between footfall and items_sold using scatter plot.

**(c)** - You notice that 80% of transactions in the dataset occurred without any promotion. Describe how this imbalance could affect your model and what steps you would take to address it.

**(Ans)** - if 80% of transactions in the dataset occurred without any promotion then model may learn to ignore the promotion effect or can be biased towards the no promo behavior. This will give wrong results towards effectivness of promotion and also may not give proper differences between different promotion types.
To solve this issue we can
1. resample by oversampling promotion cases
2. build seperate models for Promo vs Non promo
3. may add a flag is_promotion a new column as a variable


# B3. Model Evaluation and Deployment

**(a)** - You have monthly store-level data spanning three years across 50 stores. Describe how you would set up the train-test split. Why is a random split inappropriate here? Which evaluation metrics would you use, and how would you interpret each in the context of this business problem?

**(Ans)** - Since the data is time based i.e. on monthly basis so we can split the train test in 80-20 percent. For eg. if we have the data for 3 years then we can train the 2.5 year data while test the 6 months data.
In a time based dataset, a random split mixes the records from different periods, so the model may train on latermonths and test on earlier once. This will give it access to patterns that would not be known in real life at prediction time. As a result, model will give high performance but will fail when used in real world because it actually can't "see the future".
Since this is a regression problem so we can use Mean absolute Error (MAE), Root Mean squared error (RMSE) and R squared in which R-square will give the overall model fit. 
MAE will show us the how much on an average the model is off by X items per store per month.
RMSE penalizes large errors more heavily which will capture the bad promotion decisions.
R-squared will give us the % of sales variation explained by the features.

**(b)** - After training, the model recommends the Loyalty Points Bonus for Store 12 in December and the Flat Discount for Store 12 in March. Using the concept of feature importance, explain how you would investigate and communicate to the marketing team why the model makes different recommendations for the same store in different months.

**(Ans)** - we can identify the global influential feature like month or seasonality , festival indicators or promotion type etc.
else we can directly compare the sales of diffrent stores on March and December. 

Mostly in December due to festivals and holidays demand is high so customers are less price sensitive and prefers loyalty points bonus, alternatively on March since there is no reason for feastivity or holidays so customer expect flat discounts.

We can explain this to the marketing team as "In December demand is already high, so incentives that increase the customer retention work better and on the other hand in March as demand is weaker, so price based promotions are more effective in driving sales.

**(c)** - The trained model needs to generate recommendations at the start of every month for all 50 stores without being retrained each time. Describe the end-to-end deployment process: how you would save the model, how new monthly data would be prepared and fed in, and what monitoring you would put in place to detect when the model's performance has degraded and retraining is needed.

**(Ans)** - To save model we can use pickle.
At the start of each month: we have to collect data first like store attributes, planned promotions, calendar features and then can do some feature engineering
using encoding (promotion type, location), scaling and date features. Once this is done we can load model to generate the predictions using 
predictions = model.predict(new_data) and model = joblib.load("promotion_model.pkl") , then we can also add a schedular on monthly basis usings tools like Airflow etc.
At the end we have to compare predicted and actual items sold by checking the MAE over time, if the error increases then we can say that the model is degrading and so retraining is needed.
 

