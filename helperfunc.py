import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm 
from typing import Union

def generate_pc(data=pd.DataFrame()):
    """
    Generate the first principal component of a given input.
    
    Parameters:
    data (pandas.DataFrame or array-like, default=pd.DataFrame()): 
        Input data containing the features. If not a DataFrame, it will be converted to one.
    
    Returns:
    pandas.Series: First principal component of the input data.
    """
    # Convert input to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except ValueError:
            raise ValueError("Input data could not be converted to a DataFrame")

    # Check if the DataFrame is empty
    if data.empty:
        raise ValueError("Input data is empty")

    # Remove any non-numeric columns
    numeric_df = data.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("No numeric columns found in the input data")

    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Apply PCA
    pca = PCA(n_components=1)
    first_pc = pca.fit_transform(scaled_data)
    
    # Convert the result to a pandas Series
    first_pc_series = pd.Series(first_pc.flatten(), index=numeric_df.index, name="First_PC")

    return {"pc1":first_pc_series,
            "explained_var": pca.explained_variance_ratio_
    }

    # Example usage:
    # import pandas as pd
    # import numpy as np

    # # Using a DataFrame
    # df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    # first_pc_df = generate_first_principal_component(df)
    # print("From DataFrame:")
    # print(first_pc_df)

    # # Using a NumPy array
    # arr = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    # first_pc_arr = generate_first_principal_component(arr)
    # print("\nFrom NumPy array:")
    # print(first_pc_arr)

    # # Using the default empty DataFrame
    # try:
    #     generate_first_principal_component()
    # except ValueError as e:
    #     print("\nDefault empty DataFrame:")
    #     print(f"Error: {e}")

def fit_trend_cycle_model(data, column, method='ucm', lambda_param=1600):
    """
    Fit a trend-cycle decomposition model using either Unobserved Components Model (UCM)
    or Hodrick-Prescott (HP) filter.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame containing the time series data.
    column (str): Name of the column in the DataFrame to be modeled.
    method (str): Either 'ucm' for Unobserved Components Model or 'hp' for Hodrick-Prescott filter.
    lambda_param (float): Smoothing parameter for HP filter (default is 1600 for quarterly data).
    
    Returns:
    dict: A dictionary containing the decomposed components (trend and cycle).
    """
    if method not in ['ucm', 'hp']:
        raise ValueError("Method must be either 'ucm' or 'hp'")
    
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame")
    
    series = data[column]
    
    if method == 'ucm':
        # Unobserved Components Model
        ucm_model = {
            'level': 'local linear trend',
            'cycle': True,
            'damped_cycle': True,
            'stochastic_cycle': True
        }
        model = sm.tsa.UnobservedComponents(series, **ucm_model)
        results = model.fit(method='powell', disp=False)
        
        trend = results.level["smoothed"]
        cycle = results.cycle["smoothed"]*100
    
    elif method == 'hp':
        # Hodrick-Prescott Filter
        cycle, trend = sm.tsa.filters.hpfilter(series, lamb=lambda_param)
    
    return pd.DataFrame({
        'trend': trend,
        'cycle': cycle
    }, index = data.index)

# Example usage:
# import pandas as pd
# import numpy as np

# # Create a sample DataFrame with a time series
# np.random.seed(0)
# dates = pd.date_range(start='2000-01-01', periods=100, freq='Q')
# data = pd.DataFrame({
#     'date': dates,
#     'GDP': np.cumsum(np.random.normal(0.5, 0.3, 100)) + 100
# })
# data.set_index('date', inplace=True)

# # Use UCM method
# ucm_result = fit_trend_cycle_model(data, 'GDP', method='ucm')
# print("UCM Trend (first 5 values):", ucm_result['trend'].head())
# print("UCM Cycle (first 5 values):", ucm_result['cycle'].head())

# # Use HP filter method
# hp_result = fit_trend_cycle_model(data, 'GDP', method='hp', lambda_param=1600)
# print("\nHP Trend (first 5 values):", hp_result['trend'].head())
# print("HP Cycle (first 5 values):", hp_result['cycle'].head())


def rmse(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Calculate Root Mean Squared Error"""
    return np.sqrt(np.mean((forecasts - actuals) ** 2))

def mase(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Calculate Mean Absolute Scaled Error"""
    values = []
    for i in range(1, actuals.shape[0]):
        values.append(abs(actuals[i] - forecasts[i]) / (abs(actuals[i] - actuals[i - 1]) / actuals.shape[0] - 1))
    return np.mean(np.array(values))

def smape(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(2 * np.abs(forecasts - actuals) / (np.abs(actuals) + np.abs(forecasts)))

def theils_u(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Calculate Theil's U statistic"""
    numerator = np.sqrt(np.mean((forecasts - actuals) ** 2))
    denominator = np.sqrt(np.mean(actuals ** 2)) + np.sqrt(np.mean(forecasts ** 2))
    return numerator / denominator if denominator != 0 else np.inf

def mdrae(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Calculate Median Relative Absolute Error"""
    naive_forecast = np.roll(actuals, 1)
    naive_forecast[0] = actuals[0]
    abs_diff = np.abs(naive_forecast - actuals)
    
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rae = np.abs(forecasts - actuals) / abs_diff
    
    # Replace inf and nan with a large number (you can adjust this value)
    rae[np.isinf(rae) | np.isnan(rae)] = 1e9
    
    return np.median(rae[1:])  # Exclude the first element as it's not meaningful

def calculate_metrics(forecasts: list[float], actuals: list[float]) -> dict[str, float]:
    """
    Calculate various forecast accuracy metrics.
    
    Args:
    forecasts (List[float]): List of forecast values
    actuals (List[float]): List of actual values
    
    Returns:
    Dict[str, float]: Dictionary containing the calculated metrics
    """
    f_array = np.array(forecasts)
    a_array = np.array(actuals)
    
    return {
        "RMSE": rmse(f_array, a_array),
        #"MASE": mase(f_array, a_array),
        "SMAPE": smape(f_array, a_array),
        "Theil's U": theils_u(f_array, a_array),
        "MDRAE": mdrae(f_array, a_array)
    }


def add_lags(df, lags, columns=None):
    """
    Add lagged columns to a DataFrame or numpy array.
    
    Parameters:
    df (pd.DataFrame or np.ndarray): Input DataFrame or numpy array
    lags (int): Number of lags to create
    columns (list, optional): List of column names to create lags for. 
                              If None, use all columns.
    
    Returns:
    pd.DataFrame: DataFrame with added lag columns
    
    Raises:
    ValueError: If input is not a pandas DataFrame or numpy array
    """
    # Input validation
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)
    elif not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame or numpy array")
    
    df_with_lags = df.copy()
    
    # If columns are not specified, use all columns
    if columns is None:
        columns = df.columns.tolist()
    
    for col in columns:
        for lag in range(1, lags + 1):
            df_with_lags[f'{col}_{lag}'] = df_with_lags[col].shift(lag)
    
    return df_with_lags

