import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

def load_data(indices: List[int], path_template: str) -> np.ndarray:
    """
    Loads data from CSV files based on indices and a path template.
    
    Args:
        indices: List of indices to format into the path template.
        path_template: String template for file paths, e.g., "data/file_{}.csv".
        
    Returns:
        Concatenated and scaled data as a numpy array.
    """
    train_data = []
    for i in indices:
        path = path_template.format(i)
        try:
            df = pd.read_csv(path)
            # Basic validation of column count, adjust as needed for specific data
            if df.shape[1] >= 11: 
                # Assuming last columns are features if there's an index col
                # This logic might need specific adjustment based on exact CSV format
                if df.shape[1] == 12:
                     data = df.iloc[:, 1:].values
                else:
                     data = df.values
            else:
                print(f"Warning: Unexpected number of columns in {path}: {df.shape[1]}")
                continue
                
            print(f"Loaded {path} with shape {data.shape}")
            train_data.append(data)
        except FileNotFoundError:
            print(f"Error: File not found at {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    if not train_data:
        raise ValueError("No training data loaded! Please check your CSV files and paths.")
        
    return scale_to_unit_range(np.vstack(train_data))

def data_preprocessing(data: np.ndarray, n_past: int, n_future: int = 1, target_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares data for time series forecasting.
    
    Args:
        data: Input data array of shape (num_samples, num_features).
        n_past: Number of past time steps to use as input.
        n_future: Number of future time steps to predict.
        target_indices: List of column indices to use as targets. 
                        If None, defaults to columns 3 to 6 (exclusive 7).
    
    Returns:
        Tuple of (X, y) arrays.
    """
    if target_indices is None:
        # Default to u1, u2, u3 (indices 0, 1, 2)
        target_slice = slice(0, 3)
    else:
        target_slice = target_indices

    X, y = [], []
    # Vectorized approach could be faster but list comprehension is clear for sliding window
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past : i])
        # Target is n_future steps ahead? Or sequence? 
        # Original code: data[i + n_future - 1, 3:7] implies single step prediction at horizon n_future
        if isinstance(target_slice, slice):
             y.append(data[i + n_future - 1, target_slice])
        else:
             y.append(data[i + n_future - 1, target_slice])
             
    return np.array(X), np.array(y)
def scale_to_unit_range(data):
    """
    Scales each column of the input (N, 11) array to [0, 1] based on custom min/max per column.

    Parameters:
        data (np.ndarray): Input array of shape (N, 11)

    Returns:
        np.ndarray: Scaled array of shape (N, 11), each column normalized to [0, 1]
    """
    if data.shape[1] != 11:
        raise ValueError("Input array must have exactly 11 columns.")

    # Define per-column min and max values
    min_vals = np.array([0, 0, 0, 0, 0.00156, 0, 0.002, 0, 0.59545, -16.7, 0.00139])
    max_vals = np.array([1.2, 7.5, 150, 3, 1.3422, 59, 0.102, 1100, 1.3638, 33.6, 0.017921])

    # Apply min-max scaling
    scaled = (data - min_vals) / (max_vals - min_vals)

    # Clip any small numeric deviations from [0,1]
    scaled = np.clip(scaled, 0, 1)

    return scaled
def unscale_from_unit_range(scaled_data):
    """
    Unscales each column of a (N, 3) array from [0, 1] back to original value ranges.

    Parameters:
        scaled_data (np.ndarray): Input scaled array of shape (N, 3)

    Returns:
        np.ndarray: Unscaled array of shape (N, 3)
    """
    if scaled_data.shape[1] !=3:
        raise ValueError("Input array must have exactly 3 columns.")

    # Original min and max values per column
    min_vals = np.array([0,0,0])
    max_vals = np.array([1.2, 7.5, 150])
    # Reverse the min-max scaling
    unscaled = scaled_data * (max_vals - min_vals) + min_vals

    return unscaled
def scale_to_unit_range_target(data):
    """
    Scales each column of the input (N, 3) array to [0, 1] based on custom min/max per column.

    Parameters:
        data (np.ndarray): Input array of shape (N, 3)

    Returns:
        np.ndarray: Scaled array of shape (N, 3), each column normalized to [0, 1]
    """
    if data.shape[1] != 3:
        raise ValueError("Input array must have exactly 3 columns.")

    # Define per-column min and max values
    min_vals = np.array([0,0,0])
    max_vals = np.array([1.2, 7.5, 150])
    # Apply min-max scaling to bring each column to [0, 1]
    scaled = (data - min_vals) / (max_vals - min_vals)

    return scaled