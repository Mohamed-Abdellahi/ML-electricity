import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


path = '/Users/mohamedabdellahi/Desktop/Trust the process/1- M203/master thesis/data/germany-Main.xlsx'
pd.set_option('display.max_columns', None)
# Function to explore the data
def explore_data(df):
    print(df.describe())
    print(df.isnull().sum())
    print(df.dtypes)

# Function for feature engineering
def engineer_features(df):
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek # this is a panda function that give the nmber of the date (weeknd are 5, 6) Monday is 0
    df['Month'] = df['Date'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    return df


# Function for data cleaning
def clean_data(df):
    df.fillna(df.mean(), inplace=True)  # Example: fill missing values with the mean
    # Add other data cleaning steps here
    return df


# Function for data transformation
def transform_data1(df, target_column_name):
    # Drop the target column and any non-numeric columns
    # attention there are categorical data that shouldn't be scaled, so w have exclude them
    X = df.select_dtypes(include=[np.number]).drop([target_column_name], axis=1)
    y = df[target_column_name]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Convert the scaled features back to a DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled_df, y


# Assuming 'Date' column or any datetime type columns are correctly excluded from X_to_scale
def transform_data(df, target_column_name, exclude_from_scaling=None):
    if exclude_from_scaling is None:
        exclude_from_scaling = []

    # Ensure 'Date' column is also excluded if still present
    df_numeric = df.select_dtypes(include=[np.number, 'bool'])  # 'bool' included if you have boolean types
    X = df_numeric.drop([target_column_name] + exclude_from_scaling, errors='ignore', axis=1)
    y = df[target_column_name]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns) # you may ask why: After scaling, X_scaled is a NumPy array
    # without the column names, because the scaling operation returns a NumPy array. This line converts X_scaled back
    # into a pandas

    # Re-include the excluded features without scaling
    X_excluded = df[exclude_from_scaling].reset_index(drop=True)
    X_final = pd.concat([X_scaled_df, X_excluded], axis=1)
    print("Make sure transformation is well done:")
    print(X_final)
    return X_final, y


# Function to split the data
def split_data(X, y, test_size=0.2, random_state=203):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


###
def train_model(X_train, y_train, params):
    model = SVR(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    # Print the evaluation metrics
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")

    return predictions, mse, mae, rmse, r2


def tune_hyperparameters(X_train, y_train, model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_



# Main function to run all steps
def main(file_path, target_column_name):
    # Load the data
    df = pd.read_excel(file_path, skiprows=[1])

    # Call the functions one by one
    explore_data(df)
    df = engineer_features(df)
    print(df.head())
    df = clean_data(df)
            #centrer et reduire
    X_scaled, y = transform_data(df, target_column_name, ['Hour', 'DayOfWeek', 'Month', 'IsWeekend'])
            #splitting the data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    #print("after splitting")
    #print('X_train, X_test')
    #print(X_train, X_test) # validated
    #print("y_train, y_test")
    #print(y_train, y_test) # validated

      # Model training and evaluation can go here

    # let's train the model


    model = SVR()
    model.fit(X_train, y_train)

    # what do we get ?
        # metrics for accuracy
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    epsilon = 1e-8
   # mape = np.mean(np.abs((y_test - predictions) / (y_test + epsilon))) * 100 - this

    print("Mean Squared Error:", mean_squared_error(y_test, predictions))
    print("R^2 Score:", r2_score(y_test, predictions))
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
   # print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

    # Graphical presenetation
    plt.figure(figsize=(14, 7))
    plt.plot(predictions, label='Predicted', alpha=0.7)
    plt.plot(y_test.values, label='Actual', alpha=0.5)  # Ensure y_test is a series, so we access .values
    plt.title('Actual vs. Predicted Values Over Time')
    plt.xlabel('Time/Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Run the main function
main(path, 'Day Ahead Auction')
