from wine_functions import concat_dataframes, fetch_wine_data, load_red_wine_data, load_white_wine_data, \
    add_color_feature, concat_dataframes, split_dataset, CustomAttributeTransformer, create_data_pipeline
import pandas as pd

# fetch_wine_data()

red_wine = load_red_wine_data()
# print(red_wine.head())
# print(red_wine.info())
# print(red_wine["quality"].value_counts())


white_wine = load_white_wine_data()
# print(white_wine.head())
# print(white_wine.info())
# print(white_wine["quality"].value_counts())

add_color_feature(red_wine, white_wine)

# print(red_wine.head())
# print(red_wine.info())

# print(white_wine.head())
# print(white_wine.info())

wine = concat_dataframes(red_wine, white_wine)

# print(wine.info())
# print(wine.head())
# print(wine.tail())

# print(wine["color"].value_counts())

train, train_labels, test, test_lables = split_dataset(wine)

# print(train.info())
# print(train.head())
# print(train["color"].value_counts())
# print(train_labels.info())
# print(train_labels.head())
# print(train_labels.value_counts())

# print(train.tail())
# print(train_labels.tail())

# attr_adder = CustomAttributeTransformer(remove_pH=True)
# wine_extra_attribs = attr_adder.transform(train.values)

# print(wine_extra_attribs.view())

# test_view_df = pd.DataFrame(wine_extra_attribs)

# print(test_view_df.head())


cleaner = create_data_pipeline(train)
train_prepared = cleaner.transform(train)

print(train_prepared[0])