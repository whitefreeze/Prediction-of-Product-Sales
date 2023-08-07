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

## Limitations & Next Steps

While this model can be useful if implemented in a strategic manner, it would be better to try more models and see if they can perform better than the DecisionTreeRegressor using default hyperparameters. When the best model is found using default hyperparameters, then the hyperparameters of that model should then be tuned in order to maximize the predictive power of the model. 

There is currently a lot of room for improvement and it is expected that model tuning and finding a better model will yield better predictions.
