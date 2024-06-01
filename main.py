import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def read_description_file(filename):
    try:
        with open(filename, 'r') as file:
            des = file.read()
        return des
    except FileNotFoundError:
        print("Description file not found.")
        return None


# import data for file in the program
data = pd.read_csv('train.csv')

# print(data.head())
print(data.info())
print('---------------------------------------------------------------------')
# print(data.describe())
print(data.columns)
print('---------------------------------------------------------------------')

# Taking SalePrice and Separating target value
x_data = data.drop(columns=['SalePrice'])
y_data = data['SalePrice']

num_features = x_data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = x_data.select_dtypes(include=['object']).columns

num_transformer = Pipeline(steps=[
    ('inputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('inputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(x_data, y_data)

description = read_description_file("data_description.txt")
if description:
    print("Description of the model:")
    print(description)

T_data = pd.read_csv("test.csv")

predictions = model.predict(T_data)

predictions_df = pd.DataFrame({'Id': T_data['Id'], 'SalePrice': predictions})
predictions_df.to_csv("prediction.csv", index=False)

y_data_predictions = model.predict(x_data)
mse_train = mean_squared_error(y_data, y_data_predictions)
print("Mean Squared Error (Train):", mse_train)

plt.figure(figsize=(10, 6))
plt.plot(predictions, label='Predicted Sale Price', color='blue')
plt.xlabel("Data Points")
# "data points" refers to the individual instances.
plt.ylabel("Sale Price")
plt.title("Predicted Sale Prices")
plt.legend()
plt.grid(True)
plt.show()
