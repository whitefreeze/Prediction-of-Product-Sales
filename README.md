# Prediction of Product Sales
## Development of a model to predict item sales at outlets. 

**Author**: Josh Blackburn

### Business problem:

Grocery stores of various sizes want to be able to predict which new products will have the best sales so that order size, product placement, promotion, stocking, etc, can all be optimized.

### Data:
There are 8,523 different observations with 12 different features about the outlets and the products they carry.

## Methods

- No duplicate observations were found or removed.
- Inconsistent categorical data was detected and replaced with appropriately named data.
- Numeric pipeline was created with a mean imputer, for missing numeric values, and standard scaler, for modeling.
- Categorical pipeline was created with a 'most frequent' imputer, for missing categorical values, and one-hot encoded, for modeling.
- Data was preprocessed before modeling. 
- The Regression Tree Model performed the best model out of the models tested. 
- Even though it did not perform very well, its results are at least reasonable, as deteremined by looking at the MAE, RMSE and R2 scores.

## Results

#### Correlation Heatmap of Numerical Features

![viz_correlation_heatmap](https://github.com/whitefreeze/Prediction-of-Product-Sales/assets/13343127/a84998a8-3125-4626-9396-2bdfabfe8b8d)

> We can see that there is a positive and "moderate" correlation between the Item_MRP (Maximum Retail Price) and the Item_Outlet_Sales (Sales of a product in a particular store).

#### Scatter Plot: Item_Outlet_Sales vs Item_MRP

![viz_scatterplot](https://github.com/whitefreeze/Prediction-of-Product-Sales/assets/13343127/c9323693-d9ef-4d29-94eb-4a9e1b0cbece)

> This scatterplot shows that there is a positive correlation with Item MRP and Item Outlet Sales. As one increases in value, so does the other.

## Model

The best performing model to predict sales was the DecisionTreeRegressor. 

The folowing are the DecisionTreeRegressor Test Scores (metrics for the model):

> * MAE: 1,004.1214 
> * MSE: 2,184,218.0003 
> * RMSE: 1,477.9100 
> * R2: 0.2083

The model can account for about 20% of the variation in the predicted value and the average error in value from the true value Item_Outlet_Sales is very close to 1,000. Depending on the predicted sales volume, this could be useful. 

## Recommendations:

In order to take advantage of the model and increase Item_Outlet_Sales for new products, my recommendation is to only select the highest predicted performers and order 1,000 less than the prediction. Then order more as needed. For larger predicted sales (8,000 to 10,000 or higher will be better), this will allow new products to be tried, while minimizing risk of having new stock going unsold.

Due to the value of the mean average error (MAE), I would not recommend new products be tried when the predicted Item_Outlet_Sales is 1,000-2,000 or less (again, higher is better).

## Revisitation: Importances and Coefficients

### Linear Regression

A Linear Regression model has been optimized and fit to the data, giving us informative coefficients. Looking at the top three largest coefficients, we can see the following: 

![PPS_linreg_top3!](https://github.com/whitefreeze/Prediction-of-Product-Sales/assets/13343127/1525a9bd-e6d3-4ab4-bf2e-a665da55be7f)

#### Coefficients that Positively Influence Final Grade:
* **Outlet_Type_Supermarket Type3**
* > Being in the Outlet_Type_Supermarket Type3 group (being sold from that type of supermarket) increases target by 1,473.25
* **Outlet_Identifier_OUT027**
* > Being in the Outlet_Identifier_OUT027 group (being sold from that particular supermarket) increases target by 1,473.25
* **Outlet_Type_Supermarket Type1**
* > Being in the Outlet_Type_Supermarket Type1 group (being sold from that type of supermarket) increases target by 1,151.28

### Decision Tree Regressor

A Decision Tree Regressor model has been optimized and fit to the data. Using the model's built-in .feature_importances_, we can find which features the model determined were the most important in its calculations. 

Looking at the top five most important features, we can see the following: 

![PPS_dec_tree_top5!](https://github.com/whitefreeze/Prediction-of-Product-Sales/assets/13343127/9137fd36-c7dd-4f11-99a8-caa858ca47cd)

#### What the feature importances tell us:
* **Item_MRP**: is the most important feature for predicting 'Item_Outlet_Sales'.
* **Outlet_Type_Supermarket Type3**: is about four times less important than 'Item_MRP', but is still the second most important feature.
* **Item_visibility**: is the third-most important feature, but very similar to importance as second place.
* **Outlet_Type_Supermarket Type1**: is the fourth-most important feature.
* **Outlet_Type_Supermarket Type2**: is the fifth-most important feature and is about 75% the importance of the second place feature.

### SHAP Summary Plots & Interpretations

#### Feature Importances: SHAP

![PPS_SHAP_bar_top5!](https://github.com/whitefreeze/Prediction-of-Product-Sales/assets/13343127/210bd0b3-1f73-4bb9-bbcb-45c40eb5adfe)

#### Feature Importances: scikit-learn

Using scikit-learn's built-in feature importances attribute (.feature_importances_) for tree-based models.

![PPS_dec_tree_top5!](https://github.com/whitefreeze/Prediction-of-Product-Sales/assets/13343127/0c1c9d00-7391-4ced-a0d5-cac2f60824a3)

**Feature Importance:** SHAP vs scikit-learn's Decision Tree Regression Built-in Method

Comparing the top five most important features between the SHAP calculations and the scikit-learn's built-in Decision Tree Regression  method, we can see that order notwithstanding, the top five are the same, in addition to the top feature ('Item_MRP'), which is the same in both. 

#### SHAP Dot Plot

![PPS_SHAP_dot_top5!](https://github.com/whitefreeze/Prediction-of-Product-Sales/assets/13343127/735e8cbd-de1c-454b-bf23-9bb2f6e86bb4)

**Feature Importance:** SHAP Dot Plot - Top Three (3)

Let's look at the top three most important features as determined by the SHAP dot plot and see how they influence the model's predictions.
1. As 'Item_MRP' increases in value, it increases the target ('Item_Outlet_Sales') most drastically.
2. As 'Outlet_Type_Supermarket Type1' increases in value, it also drastically increases the target ('Item_Outlet_Sales'). 
3. As 'Outlet_Type_Supermarket Type3' increases, just like the top two, it also strongly increases the value of the target. 


### SHAP Force Plot/LIME Tabular Explanations & Interpretations

For the both the SHAP Force Plot and the LIME explanation, we have selected two specific rows to dissect: 
1. The row with the item having the highest target ('Item_Outlet_Sales') value.
2. The row with the item having the lowest target ('Item_Outlet_Sales') value.

We chose these two observations to gain insight as two why they had the most extreme target values.

#### LIME Tabular Explanation Plot: Minimum Target Value

![PPS_LIME_target_min](https://github.com/whitefreeze/Prediction-of-Product-Sales/assets/13343127/af42f78f-a7ae-48ac-acf3-7d45f86bee69)

**Feature Importance:** Minimum Target Value Explanation

From the visualization above, we can see that for the target's minimum value observation, these were the features that most heavily influenced the low predictions, according to LIME: 
* The item was NOT sold from any of the 'Outlet_Type_Supermarket Types' (1,2 or 3): a negative influence (-).
* The Item_MRP was less than 99.43: a negative influence (-).
* The other factors that most heavily influenced the prediction is type of food category that it was not a part of:
> * NOT a Household Item: a negative influence (-).
> * NOT a Starchy Food Item: a negative influence (-).
> * NOT Bread-type Item: a negative influence (-).
> * NOT Seafood Item: a negative influence (-).
> * NOT Fruits & Vegetables Item: a negative influence (-).
> * NOT an item of type 'Others': a POSITIVE influence (+).

#### LIME Tabular Explanation Plot: Maximum Target Value

![PPS_LIME_target_max](https://github.com/whitefreeze/Prediction-of-Product-Sales/assets/13343127/4de93696-20e4-45b1-ab95-62ccffd04e9c)

**Feature Importance:** Maximum Target Value Explanation

From the visualization above, we can see that for the target's maximum value observation, these were the features that most heavily influenced the high predictions, according to LIME: 
* The item was NOT sold from 'Outlet_Type_Supermarket Types' (2 or 3): a negative influence (-).
* The item WAS sold from 'Outlet_Type_Supermarket Type1': a POSITIVE influence (+).
* The Item_MRP was higher than 183.59: a POSITIVE influence (+).
* The outlet that the item was sold from was NOT 'Outlet_Identifier_OUT0027' (the highest performing outlet, which can be seen in the 'Future Data from Analysis Option' Section at the end of this notebook).
* The other factors that most heavily influenced the prediction is type of food category that it was not a part of:
> * NOT a Hard Drinks Item: a negative influence (-).
> * NOT an item of type 'Others': a POSITIVE influence (+).
> * NOT a Meat Item: a negative influence (-).
> * NOT a Household Item: a negative influence (-).
> * NOT a Bread Item: a POSITIVE influence (+).

#### SHAP Force Plot: Minimum Target Value

![PPS_SHAP_target_min](https://github.com/whitefreeze/Prediction-of-Product-Sales/assets/13343127/7ac81b4d-89e2-481a-b497-a429caa8cf15)

**Feature Importance:** Minimum Target Value Explanation

From the Force Plot visualization above, we can see that for the target's minimum value observation, these were the features that most heavily influenced the low predictions, according to the SHAP Force Plot: 
* The Item_MRP was 45.14: a negative influence (-).
* The item was NOT sold from any of the 'Outlet_Type_Supermarket Types' (1,2 or 3): a negative influence (-).
* The outlet was NOT Outlet_Identifier_OUT018': a negative influence (-).

No positive influences (+) were significant enough to have a label in the visualization. 

#### SHAP Force Plot: Maximum Target Value

![PPS_SHAP_target_max](https://github.com/whitefreeze/Prediction-of-Product-Sales/assets/13343127/c107e6a0-529e-40d7-a807-4f3f72a7a6f6)

**Feature Importance:** Maximum Target Value Explanation

From the Force Plot visualization above, we can see that for the target's maximum value observation, these were the features that most heavily influenced the high predictions, according to the SHAP Force Plot: 
* The Item_MRP was 261.3: a POSITIVE influence (+).
* The Item_Visibility was 0.0195: a POSITIVE influence (+).
* The Outlet_Size was 2: a POSITIVE influence (+).
* The item WAS sold from 'Outlet_Type_Supermarket Type1': a POSITIVE influence (+).
* The Item_Fat_Content_Regular was 1 (not low-fat): a POSITIVE influence (+).

No negative influences (-) were significant enough to have a label in the visualization. 


## Limitations & Next Steps

While this model can be useful if implemented in a strategic manner, it would be better to try more models and see if they can perform better than the DecisionTreeRegressor using default hyperparameters. When the best model is found using default hyperparameters, then the hyperparameters of that model should then be tuned in order to maximize the predictive power of the model. 

There is currently a lot of room for improvement and it is expected that model tuning and finding a better model will yield better predictions.
