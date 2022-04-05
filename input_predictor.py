
import joblib
import pandas as pd

# loads the saved predictor, and then prompts the user for the 12 features of a wine
# runs them through the predictor, and prints the predicted quality

final_model = joblib.load("final_model.pkl")
cleaner = joblib.load("cleaner.pkl")

def input_predictor():

    fixed_acidity = float(input("Enter fixed acidity:"))
    volatile_acidity = float(input("Enter volatile acidity:"))
    citric_acid = float(input("Enter citric acid:"))
    residual_sugar = float(input("Enter residual sugar:"))
    chlorides = float(input("Enter chlorides:"))
    free_sulfur_dioxide = float(input("Enter free sulfur dioxide:"))
    total_sulfur_dioxide = float(input("Enter total sulfur dioxide:"))
    density = float(input("Enter density:"))
    pH = float(input("Enter pH:"))
    sulphates = float(input("Enter sulphates:"))
    alcohol = float(input("Enter alcohol:"))
    color = float(input("Enter color:"))

    df = pd.DataFrame({'fixed_acidity': [fixed_acidity],
                       'volatile_acidity': [volatile_acidity],
                       'citric_acid': [citric_acid],
                       'residual_sugar': [residual_sugar],
                       'chlorides': [chlorides],
                       'free_sulfur_dioxide': [free_sulfur_dioxide],
                       'total_sulfur_dioxide': [total_sulfur_dioxide],
                       'density': [density],
                       'pH': [pH],
                       'sulphates': [sulphates],
                       'alcohol': [alcohol],
                       'color': [color],})

    data_prepared = cleaner.transform(df)

    print(final_model.predict(data_prepared))

input_predictor()