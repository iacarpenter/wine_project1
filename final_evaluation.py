from wine_functions import concat_dataframes, fetch_wine_data, load_red_wine_data, load_white_wine_data, \
    add_color_feature, concat_dataframes, split_dataset, create_data_pipeline
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

red_wine = load_red_wine_data()
white_wine = load_white_wine_data()
add_color_feature(red_wine, white_wine)
wine = concat_dataframes(red_wine, white_wine)

train, train_labels, test, test_labels = split_dataset(wine)

cleaner = create_data_pipeline(train)

joblib.dump(cleaner, "cleaner.pkl")

final_model = joblib.load("final_model.pkl")

test_prepared = cleaner.transform(test)

final_predictions = final_model.predict(test_prepared)
final_mse = mean_squared_error(test_labels, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
# final RMSE of around 0.62