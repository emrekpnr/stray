from typing import Literal

import numpy as np


def normalize_data(data: np.ndarray, method: Literal["unitize", "standardize"] = "unitize") -> np.ndarray:
    normalized_columns = []

    for column in data.T:
        if method == "unitize":
            min_column = np.min(column)
            max_column = np.max(column)
            if max_column != min_column:
                normalized = (column - min_column) / (max_column - min_column)
            else:
                normalized = np.zeros_like(column)
        else:
            column_median = np.median(column)
            iqr = np.percentile(column, 75) - np.percentile(column, 25)
            normalized = (column - column_median) / iqr
        normalized_columns.append(normalized)

    return np.array(normalized_columns).T
