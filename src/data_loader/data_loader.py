import pandas as pd


def load_data(path:str) -> list[pd.DataFrame, tuple[int], int]:
    '''
    Loads the dataset, removes the non-numeric values

    Returns clean data, its shape and number of non-valid rows
    '''
    data = pd.read_csv(path)
    data_numeric = data.apply(pd.to_numeric, errors='coerce')
    clean_data = data_numeric.dropna()

    new_data_shape = clean_data.shape

    non_numeric_rows = data_numeric[data_numeric.isna().any(axis=1)]    
    num_nonvalid_rows = non_numeric_rows.shape[0]

    return[clean_data, new_data_shape, num_nonvalid_rows]
