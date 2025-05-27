# Stray (Python): Anomaly Detection in High-Dimensional Data

This repository contains a Python re-implementation of the **Stray** algorithm for detecting anomalies in high-dimensional datasets. The original R implementation is available at:

🔗 https://github.com/pridiltal/stray/tree/master

## 📌 Features

- Detects both global (isolated) and local (clustered) outliers
- Fully unsupervised, with thresholding based on Extreme Value Theory (EVT)
- Uses k-nearest neighbor distances with maximum gap scoring
- Compatible with real-world and synthetic datasets

## 📁 File Overview

| File                    | Description                                                        |
|-------------------------|--------------------------------------------------------------------|
| `find_HDOutliers.py`    | Core anomaly detection logic using k-NN and anomaly scoring        |
| `find_threshold.py`     | Implements EVT-based thresholding to label anomalies               |
| `display_HDOutliers.py` | Utility for visualizing the anomalies in 2D                        |
| `main.py`               | Example pipeline to run the algorithm on a dataset                 |
| `calculate_fpr.py`      | Computes false positive rates for different thresholds             |

## ⚙️ Setup

### Requirements

Install dependencies via pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Usage
You can change the dataset in `main.py` to test with your own data and run main.py to execute the anomaly detection pipeline.


## 📚 References

### Tools and Libraries

- **NumPy** – https://numpy.org/
- **Pandas** – https://pandas.pydata.org/
- **Scikit-learn** – https://scikit-learn.org/
- **Matplotlib** – https://matplotlib.org/
- **Seaborn** – https://seaborn.pydata.org/
- **pyreadr** – https://github.com/ofajardo/pyreadr

### Academic References

- Priyanga Dilini Talagala, Rob J Hyndman, and Kate Smith-Miles. 2021. Anomaly detection in high-dimensional data. Journal of Computational and Graphical Statistics 30, 2 (2021), 360–374.
- Original R Implementation: https://github.com/pridiltal/stray