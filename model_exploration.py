from wine_functions import concat_dataframes, fetch_wine_data, load_red_wine_data, load_white_wine_data, \
    add_color_feature, concat_dataframes, split_dataset, create_data_pipeline, display_scores
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import joblib

# fetch_wine_data()

red_wine = load_red_wine_data()
white_wine = load_white_wine_data()
add_color_feature(red_wine, white_wine)
wine = concat_dataframes(red_wine, white_wine)

train, train_labels, test, test_labels = split_dataset(wine)

cleaner = create_data_pipeline(train)
train_prepared = cleaner.transform(train)

# linear regression model:

lin_reg = LinearRegression()
lin_reg.fit(train_prepared, train_labels)

'''
# testing that things are working correctly

some_data = train.iloc[:5]
some_labels = train_labels.iloc[:5]
some_data_prepared = cleaner.transform(some_data)
print("Predictions", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

lin_predictions = lin_reg.predict(train_prepared)
lin_mse = mean_squared_error(train_labels, lin_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
'''

lin_scores = cross_val_score(
    estimator=lin_reg,
    X=train_prepared,
    y=train_labels,
    scoring='neg_mean_squared_error',
    cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

print("\nLinear regression cross validation:")
display_scores(lin_rmse_scores)

# decision tree regressor model:

tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_prepared, train_labels)

tree_scores = cross_val_score(
    estimator=tree_reg,
    X=train_prepared,
    y=train_labels,
    scoring='neg_mean_squared_error',
    cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)

print("\nDecision tree regressor cross validation:")
display_scores(tree_rmse_scores)

# random forest regressor model:

forest_reg = RandomForestRegressor()
forest_reg.fit(train_prepared, train_labels)

forest_scores = cross_val_score(
    estimator=forest_reg,
    X=train_prepared,
    y=train_labels,
    scoring='neg_mean_squared_error',
    cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

print("\nRandom forest regressor cross validation:")
display_scores(forest_rmse_scores)

# support vector regression model:

sv_reg = SVR()
sv_reg.fit(train_prepared, train_labels)

sv_scores = cross_val_score(
    estimator = sv_reg,
    X=train_prepared,
    y=train_labels,
    scoring='neg_mean_squared_error',
    cv=10)
sv_rmse_scores = np.sqrt(-sv_scores)

print("\nSupport vector regression cross validation:")
display_scores(sv_rmse_scores)


# forest regressor grid search:

'''
forest_param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
# best params: {'max_features': 2, 'n_estimators': 30}
'''

'''
forest_param_grid = [
    {'n_estimators': [30, 60, 120], 'max_features': [1, 2, 4, 6]},
    {'bootstrap': [False], 'n_estimators': [30, 60, 120], 'max_features': [1, 2, 4]},
]
# best params: {'bootstrap': False, 'max_features': 2, 'n_estimators': 120}
'''

forest_param_grid = [
    {'bootstrap': [False], 'n_estimators': [120, 240, 480], 'max_features': [1, 2]},
]
# best params: {'bootstrap': False, 'max_features': 2, 'n_estimators': 240}

forest_reg = RandomForestRegressor()

forest_grid_search = GridSearchCV(
    estimator=forest_reg, 
    param_grid=forest_param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    refit=True,
    return_train_score=True,
    n_jobs=-1)
forest_grid_search.fit(train_prepared, train_labels)

print("\nRandom forest regressor best params:\n", forest_grid_search.best_params_)

# support vector regression grid search:

sv_param_grid = [
    {'kernel': ['rbf'], 'gamma': ['scale', 'auto']},
    {'kernel': ['linear']},
    {'kernel': ['poly'], 'degree': [3, 4, 5]},
]
# best params: {'gamma': 'scale', 'kernel': 'rbf'} (which is the default)

sv_reg = SVR()

sv_grid_search = GridSearchCV(
    estimator=sv_reg, 
    param_grid=sv_param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    refit=True,
    return_train_score=True,
    n_jobs=-1)
sv_grid_search.fit(train_prepared, train_labels)

print("\nSupport vector regression best params:\n", sv_grid_search.best_params_)


# evaluating random forest regressor with best hyperparameters:

improved_forest_scores = cross_val_score(
    estimator=forest_grid_search,
    X=train_prepared,
    y=train_labels,
    scoring='neg_mean_squared_error',
    cv=10)
improved_forest_rmse_scores = np.sqrt(-improved_forest_scores)

print("\nImproved random forest regressor cross validation:")
display_scores(improved_forest_rmse_scores)

final_model = forest_grid_search
joblib.dump(forest_grid_search, "final_model.pkl")

"""
I first evaluated four models using their default parameters with cross validation,
and chose the two with the lowest mean RMSE (which were the Random Forest Regressor
and Support Vector Regression models) to explore further with hyperparameter tuning
through grid search. I did several rounds of changing the grid search hyperparameters
for the Random Forest Regressor, and in the first two rounds the best n_estimators 
value was the maximum until a third round (with a best value of 240). Using grid search
with the hyperparameter options that I chose for the SVR gave its default hyperparameters 
as the best. Since the RFR with the improved hyperparameters performed better than the
other models including the default SVM I chose this improved RFR as the model to go 
forward with.
"""