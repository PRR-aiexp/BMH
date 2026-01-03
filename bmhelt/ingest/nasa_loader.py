import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_battery_mat(mat_path):
    mat = loadmat(mat_path)
    key = [k for k in mat.keys() if k.startswith("B")][0]
    return key, mat[key]

def to_array(x):
    """
    Convert a scalar or array to a flat 1D numpy array for pandas.
    """
    import numpy as np
    if np.isscalar(x):
        return np.array([x])       # wrap scalar in 1D array
    else:
        return np.array(x).flatten()  # flatten multi-D array


def unwrap_data(data):
    """
    Recursively unwrap nested 1x1 numpy arrays inside a cycle's data field.
    Returns the actual struct with fields like Time, Voltage_measured, etc.
    """
    import numpy as np
    while isinstance(data, np.ndarray):
        if data.size == 1:
            data = data[0]
        else:
            break
    return data


def extract_cycle(cycle, battery_id, cycle_index):
    import numpy as np

    cycle_type = cycle['type'][0]

    # unwrap nested data fields
    data = unwrap_data(cycle['data'])

    # universal field mapping
    field_map = {
        "time": ["Time", "time"],
        "voltage": ["Voltage_measured", "voltage"],
        "current": ["Current_measured", "current"],
        "temperature": ["Temperature_measured", "temperature"]
    }

    df_dict = {}
    for key, candidates in field_map.items():
        for c in candidates:
            if c in data.dtype.names:
                df_dict[key] = to_array(data[c][0])  # <- always 1D now
                break
        else:
            df_dict[key] = np.array([np.nan])      # missing field

    df = pd.DataFrame(df_dict)
    df["battery_id"] = battery_id
    df["cycle"] = cycle_index
    df["cycle_type"] = cycle_type

    return df

def unwrap_battery(battery):
    """
    Safely unwrap the actual MATLAB struct from a NumPy array.
    Works for:
      - 2D arrays like (1,1)
      - 1D arrays like (1,)
      - already unwrapped objects
    """
    if isinstance(battery, np.ndarray):
        if battery.ndim == 2:
            return battery[0,0]
        elif battery.ndim == 1:
            return battery[0]
        else:
            raise ValueError(f"Unexpected battery ndim: {battery.ndim}")
    else:
        return battery


def unwrap_cycle(cycle):
    """
    Unwrap one cycle object from nested 1x1 numpy arrays.
    """
    import numpy as np
    while isinstance(cycle, np.ndarray):
        if cycle.size == 1:
            cycle = cycle[0]
        else:
            break
    return cycle


def cycles_to_long_df(battery, battery_id):
    battery_struct = unwrap_battery(battery)
    cycles_array = battery_struct['cycle']

    dfs = []
    for i in range(cycles_array.shape[1]):  # second axis has cycles
        cycle = cycles_array[0, i]
        cycle = unwrap_cycle(cycle)
        dfs.append(extract_cycle(cycle, battery_id, i))

    return pd.concat(dfs, ignore_index=True)


def ingest_battery(mat_path):
    battery_id, battery = load_battery_mat(mat_path)
    return cycles_to_long_df(battery, battery_id)


