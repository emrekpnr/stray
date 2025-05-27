import pyreadr
import os

from display_HDOutliers import display_HDOutliers
from find_HDOutliers import find_HDOutliers


def rda_reader(file_path):
    result = pyreadr.read_r(file_path)
    return result


def main(data_name):
    file_path = os.path.join('data', f'{data_name}.rda')
    result = rda_reader(file_path)

    for key, df in result.items():
        print(f"Processing object: {key}")
        # Select only numeric columns
        numeric_df = df.select_dtypes(include='number')
        # Run HD outlier detection
        out = find_HDOutliers(numeric_df, knn_search_type='brute')
        # Visualize results
        display_HDOutliers(numeric_df, out, save_path=f'output/{data_name}.png')


if __name__ == "__main__":
    # You can change this to any other dataset name, e.g., "data_b", "data_c", etc.
    main("data_a")
