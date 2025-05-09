import gdown
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
from numpy.typing import NDArray

def download_from_gdrive(file_path: str = 'data/raw/sudoku.csv', file_id: str = '12c_UTy7pXdzJkuL1HfVdaTP15Z8QZgA2') -> None:
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print("Downloading dataset from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)
    else:
        print("File already exists:", file_path)

def get_data(file_path: str = 'data/raw/sudoku.csv') -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    download_from_gdrive(file_path)
    df = pd.read_csv(file_path)

    # Placeholder preprocessing: assume 81 features and 1 label column
    X = df.drop(columns=['label']).values.astype(np.float64)
    y = df['label'].values.astype(np.float64)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    download_from_gdrive()