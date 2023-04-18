import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


# Load data and create target and features
def csv_to_targetandfeatures(path: str = "/path/to/csv/"):
    """
        This function takes a path string to a CSV file, loads it into
        a Pandas DataFrames, splits the columns into a target column
        and a set of predictor variables, i.e. X & y.
        These two splits of the data will be used to train a supervised
        machine learning model.

        :param      path (optional): str, relative path of the CSV file

        :return     X: pd.DataFrame
                    y: pd.Series
    """
    df = pd.read_csv(f"{path}")
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    target = "estimated_stock_pct"

    # Check to see if the target variable is present in the data
    if target not in df.columns:
        raise Exception(f"Target: {target} is not present in the data")

    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# Train algorithm and return mean absolute error
def train_algorithm_with_cross_validation(X: pd.DataFrame = None, y: pd.Series = None):
    """
        This function takes the predictor and target variables and
        trains a Random Forest Regressor model. Using cross-validation,
        performance metrics will be output for each fold during training
        and an average of the values will be returned as output.

        :param      X: pd.DataFrame, predictor variables
        :param      y: pd.Series, target variable

        :return     neg_mean_absolute_error
    """
    model = RandomForestRegressor(n_estimators=150)

    # calculating mean absolute error using cross validation
    cv = cross_val_score(X, y, scoring='neg_mean_absolute_error', cv = 5)
    return np.mean(cv)
