from typing import Union

import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

matplotlib.use('TkAgg')  # Force a standard backend before pyplot import


def display_HDOutliers(data: Union[np.ndarray, pd.DataFrame], out: dict, show_plot=True, save_path: str = None):
    # Check if the input data is a numpy array or pandas DataFrame
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    dimension = data.shape[1]

    data['outcon'] = out['type']

    pyplot.figure(figsize=(6, 6))
    if dimension == 1:
        data['index'] = 0
        # Create a scatter plot for 1D data
        sns.scatterplot(
            x=data.iloc[:, 0], y=data['index'], hue=data['outcon'], style=data['outcon'],
            palette={'outlier': 'red', 'typical': 'black'}, markers={'outlier': '^', 'typical': 'o'}
        )
        # Add jitter to the y-axis to separate points
        pyplot.xlabel("Value")
        pyplot.yticks([])
        pyplot.ylabel("")
    elif dimension == 2:
        # Create a scatter plot for 2D data
        sns.scatterplot(
            x=data.iloc[:, 0], y=data.iloc[:, 1], hue=data['outcon'], style=data['outcon'],
            palette={'outlier': 'red', 'typical': 'black'}, markers={'outlier': '^', 'typical': 'o'}
        )
        # Set axis labels for 2D data
        pyplot.xlabel("Variable 1")
        pyplot.ylabel("Variable 2")
    else:
        # Create a scatter plot for higher dimensions using PCA
        pca = PCA(n_components=2)
        # Normalize the data for PCA
        principal_components = pca.fit_transform(data.iloc[:, :dimension])
        # Create a DataFrame for the PCA results
        data['PC1'] = principal_components[:, 0]
        data['PC2'] = principal_components[:, 1]
        # Create a scatter plot for the first two principal components
        sns.scatterplot(
            x=data['PC1'], y=data['PC2'], hue=data['outcon'], style=data['outcon'],
            palette={'outlier': 'red', 'typical': 'black'}, markers={'outlier': '^', 'typical': 'o'}
        )
        # Set axis labels for PCA plot
        pyplot.xlabel("PC 1")
        pyplot.ylabel("PC 2")

    # Set plot aesthetics
    # pyplot.gca().set_aspect('equal', adjustable='box')
    pyplot.title("Stray (Python) Visualization")
    pyplot.legend(title="Type")
    pyplot.tight_layout()
    # Save the plot if a save path is provided
    if save_path:
        pyplot.savefig(save_path, dpi=300)
    if show_plot:
        pyplot.show()
