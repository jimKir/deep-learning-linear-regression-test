import pandas as pd
import numpy as np

main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'  # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print(data.columns)
print(data[['SalePrice']].describe())
print(data[['LotArea']].describe())
print(data[['YearBuilt']].describe())
print(data[['GarageArea']].describe())
print(data[['MasVnrArea']].describe())
print(data[['MSZoning']].head(10))
y = data.SalePrice

predictor_cols = ['MSSubClass', 'Street', 'Alley', 'ExterQual', 'HalfBath', 'EnclosedPorch', 'TotalBsmtSF',
                  'LandContour', 'GarageCars', 'Fireplaces', 'KitchenQual', 'ExterCond', 'HeatingQC', 'YearRemodAdd',
                  'OverallCond', 'MSZoning', 'RoofStyle', 'MasVnrArea', 'LotArea', 'YearBuilt', 'GarageArea',
                  'FullBath', '1stFlrSF', '2ndFlrSF', 'TotRmsAbvGrd', 'GrLivArea', 'BedroomAbvGr']
x = data[predictor_cols]
print(x.head(2))
# categoricals handled with one-hot-encoding
x = pd.get_dummies(x)
print(x.head(2))
data = data.reset_index()

# MasVnrArea has missing values
# make copy to avoid changing original data (when Imputing)
# Redifine x to have imputed data


from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, test_X, train_y, test_y = train_test_split(x, y, random_state=0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


def score_dataset(X_train, X_test, y_train, y_test):
    preds = get_dataset(X_train, X_test, y_train, y_test)
    return mean_absolute_error(y_test, preds)


def get_dataset(X_train, X_test, y_train, y_test):
    # Define model
    model = RandomForestRegressor()
    # Fit model
    model.fit(X_train, y_train)
    # Predict
    preds = model.predict(pd.DataFrame(X_test).fillna(X_test.mean()).as_matrix().astype(np.float))
    return preds


def get_datasetXGBoost(X_train, X_test, y_train, y_test):
    # Define model
    my_model = XGBRegressor(
        n_estimators=1000,
        num_class=221000,
        eval_metric="mlogloss",
        objective="multi:softprob")
    # Fit model
    my_model.fit(X_train, y_train)

    # Predict
    preds = my_model.predict(pd.DataFrame(X_test).fillna(X_test.mean()).as_matrix().astype(np.float))
    return preds


def get_datasetXGBoostNoParams(X_train, X_test, y_train, y_test):
    # Define model
    model = XGBRegressor()
    # Fit model
    model.fit(X_train, y_train)
    # Predict
    preds = model.predict(pd.DataFrame(X_test).fillna(X_test.mean()).as_matrix().astype(np.float))
    return preds


# Read the test data
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
real_test_X = pd.get_dummies(test[predictor_cols])

from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train_plus = train_X.copy()
imputed_X_test_plus = test_X.copy()
imputed_real_test_X_plus = real_test_X.copy()

cols_with_missing = (col for col in train_X.columns
                     if train_X[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
    imputed_real_test_X_plus[col + '_was_missing'] = imputed_real_test_X_plus[col].isnull()

# Imputation to handle empty cells
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)
imputed_real_test_X_plus = my_imputer.transform(imputed_real_test_X_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed with random forest:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, train_y, test_y))

from sklearn.metrics import mean_absolute_error

# Predict multiple times with random forest
forest_preds = get_dataset(imputed_X_train_plus, imputed_X_test_plus, train_y, test_y)
print(mean_absolute_error(test_y, forest_preds))
forest_preds = get_dataset(imputed_X_train_plus, imputed_X_test_plus, train_y, test_y)
print(mean_absolute_error(test_y, forest_preds))
forest_preds = get_dataset(imputed_X_train_plus, imputed_X_test_plus, train_y, test_y)
print(mean_absolute_error(test_y, forest_preds))
forest_preds = get_dataset(imputed_X_train_plus, imputed_X_test_plus, train_y, test_y)
print(mean_absolute_error(test_y, forest_preds))

# XGBoost no params
predictions = get_datasetXGBoostNoParams(imputed_X_train_plus, imputed_X_test_plus, train_y, test_y)
from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error of XGBoost with no params: " + str(mean_absolute_error(predictions, test_y)))

# XGBoost
predictions = get_datasetXGBoost(imputed_X_train_plus, imputed_X_test_plus, train_y, test_y)
from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

# Read the test data
# test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
# test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = get_datasetXGBoost(imputed_X_train_plus, imputed_real_test_X_plus, train_y, test_y)
# We will look at the predicted prices to ensure we have something sensible.



my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
print(predicted_prices)
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


