import numpy as np
import pandas as pd
import sklearn
from pandas import DataFrame
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from scipy.stats import expon, reciprocal
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import scipy.stats as sp
from sklearn.ensemble import RandomForestRegressor

# Function to explore the data
def explore_data(df):

    print(df.describe())
    print(df.isnull().sum())
    print(df.dtypes)

# Function for feature engineering
def engineer_features1(df):
    df.set_index('Date', inplace= True)
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek # this is a panda function that give the nmber of the date (weeknd are 5, 6) Monday is 0
    df['Month'] = df['Date'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def engineer_features(df):
    # If 'Date' is not the index, set it as the index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Extract time-based features from the index
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek  # pandas function that gives the number of the day of the week (0 is Monday)
    df['Month'] = df.index.month
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)  # 5 and 6 correspond to Saturday and Sunday
    
    return df


# Function for data cleaning
def clean_data(df):
    df.fillna(df.mean(), inplace=True)  # Example: fill missing values with the mean
    # Add other data cleaning steps here
    return df


def show_correlation(df, target_variable):
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Select correlations related to the target variable
    target_corr = corr_matrix[[target_variable]].sort_values(by=target_variable, ascending=False)

    # Create a figure
    plt.figure(figsize=(8, 10))

    # Plot heatmap of the sorted target variable correlations
    sns.heatmap(target_corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title(f'Feature Correlation with {target_variable}')
    plt.show()

def show_correlation1(df, target_variable, method='spearman'):
    # Calculate the Spearman rank correlation matrix
    if method == 'pearson': 
        corr_matrix = df.corr(method=method)
    elif method == 'spearman':
        corr_matrix = sp.stats.spearmanr(df).correlation
    # Generate a mask for the upper triangle (optional, for aesthetics)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('Feature Correlation (Spearman)')
    plt.show()



# Assuming 'Date' column or any datetime type columns are correctly excluded from X_to_scale
def transform_data1(df, target_column_name, exclude_from_scaling=None):
    if exclude_from_scaling is None:
        exclude_from_scaling = ['Date']

    # Ensure 'Date' column is also excluded if still present
    #df_numeric = df.select_dtypes(include=[np.number, 'bool'])  # 'bool' included if you have boolean types
    X = df.drop([target_column_name] + exclude_from_scaling, errors='ignore', axis=1) # exclude y
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

def transform_data(df, target_column_name, exclude_from_scaling=None):
    if exclude_from_scaling is None:
        exclude_from_scaling = []

    # Set 'Date' as the index if it's not already, assuming 'Date' is the name of your datetime column
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)

    # Ensure 'Date' column is excluded from scaling by not adding it to X
    X = df.drop([target_column_name] + exclude_from_scaling, errors='ignore', axis=1)
    y = df[target_column_name]

    scaler = StandardScaler()
    # Only scale columns that are not excluded
    X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.select_dtypes(include=[np.number]).columns, index=X.index)

    # Re-include the excluded features without scaling
    X_excluded = df[exclude_from_scaling]
    X_final = pd.concat([X_scaled_df, X_excluded], axis=1)

    print("Make sure transformation is well done:")
    print(X_final.head())
    return X_final, y


# Function to split the data, but this shouldn't be applied for time series as it supposes the variables as random
def split_data1(X, y, test_size=0.2, random_state=203):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def split_data(X, y, dates, test_size=0.2):
    split_point = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    dates_train, dates_test = dates[:split_point], dates[split_point:]
    return X_train, X_test, y_train, y_test, dates_train, dates_test


def time_series_split(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_indices = []
    test_indices = []
    X_train, X_test, y_train, y_test = None, None, None, None

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Collect all train and test indices
        train_indices.append(train_index)
        test_indices.append(test_index)

    return X_train, X_test, y_train, y_test, train_indices, test_indices


def evaluate_model(model, X_test, y_test):
    """

    :type model: this is a trained model, and it's coming from the returned value of train_model
    """
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


def perform_random_search(X_train, y_train, model, param_dist, n_iter=100):
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                       n_iter=n_iter, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=203)
    random_search.fit(X_train, y_train)
    return random_search.best_params_, random_search.best_score_


def tune_hyperparameters(X_train, y_train, model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_



def plotting1(predictions, y_test):
    plt.figure(figsize=(14, 7))
    plt.plot(predictions, label='Predicted', alpha=0.7)
    plt.plot(y_test.values, label='Actual', alpha=0.5)  # Ensure y_test is a series, so we access .values
    plt.title('Actual vs. Predicted Values Over Time')
    plt.xlabel('Time/Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def plotting(predictions, y_test, dates):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, predictions, label='Predicted', alpha=0.7)  # Use dates on the x-axis
    plt.plot(dates, y_test, label='Actual', alpha=0.5)  # Ensure y_test is plotted with dates
    plt.title('Actual vs. Predicted Values Over Time')
    plt.xlabel('Date')
    plt.ylabel('Day Ahead Auction Price')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate date labels for better visibility
    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.show()
'''
here we had problems in plotting. this plotting method isn't adapted for time series split and also we weren't indexing the dates
'''


def plot_all_folds(model, X, y, test_indices):
    plt.figure(figsize=(14, 7))
    for index in test_indices:
        # Make predictions for each test set
        X_test = X.iloc[index]
        y_test = y.iloc[index]
        predictions = model.predict(X_test)

        # Plot
        plt.plot(X_test.index, predictions, label='Predicted', alpha=0.7)
        plt.plot(X_test.index, y_test, label='Actual', alpha=0.5)
    
    plt.title('Actual vs. Predicted Values Over Time')
    plt.xlabel('Date')
    plt.ylabel('Day Ahead Auction Price')
    plt.legend()
    plt.show()



def plot_training_testing(model, X, y, train_indices, test_indices):
    plt.figure(figsize=(15, 8))

    # Configure the x-axis to better handle dates
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Plot a subset or downsample
    subset_size = 1000  # Adjust as needed for visibility
    for train_idx in train_indices:
        X_train = X.iloc[train_idx].iloc[-subset_size:]
        y_train = y.iloc[train_idx].iloc[-subset_size:]
        plt.plot(X_train.index, y_train, 'b-', label='Training data' if train_idx is train_indices[0] else "", alpha=0.1)
    
    for test_idx in test_indices:
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        predictions = model.predict(X_test)
        plt.plot(X_test.index, y_test, 'g-', label='Actual Test' if test_idx is test_indices[0] else "", alpha=0.5)
        plt.plot(X_test.index, predictions, 'r--', label='Predicted Test' if test_idx is test_indices[0] else "", alpha=0.5)

    plt.title('Model Performance on Training and Testing Data')
    plt.xlabel('Date')
    plt.ylabel('Day Ahead Auction Price')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate date labels for better visibility
    plt.tight_layout()  # Adjust layout to ensure no label cut-off
    plt.show()


# this plotting only show the last testing phase, which the only interesting part 
def plot_last_test_set_predictions(model, X, y, train_indices, test_indices):
    # Get the last test index
    last_test_index = test_indices[-1]

    # Extract the last test set
    X_test = X.iloc[last_test_index]
    y_test = y.iloc[last_test_index]

    # Predict on the last test set
    predictions = model.predict(X_test)

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(X_test.index, predictions, label='Predicted', alpha= 0.7)
    plt.plot(X_test.index, y_test, label='Actual', alpha= 0.5)
    

    # Format the x-axis to handle dates better
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Auto-rotates the date labels

    plt.title('Actual vs. Predicted Values on the Last Testing Set')
    plt.xlabel('Date')
    plt.ylabel('Day Ahead Auction Price')
    plt.legend()
    plt.tight_layout()  # Adjust layout to ensure no label cut-off
    plt.show()
# Make sure to use the correct model, features 'X_scaled', target 'y', and the indices 'train_indices', 'test_indices


