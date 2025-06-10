import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from numpy.typing import NDArray

def get_data(file: str, nrows: int = None) -> Tuple[
    NDArray[np.float32], NDArray[np.float32],
    NDArray[np.int64], NDArray[np.int64]
]:
    data = pd.read_csv(file, nrows=nrows)

    feat_raw = data['quizzes']
    label_raw = data['solutions']

    feat_list: list[NDArray[np.int32]] = []
    label_list: list[NDArray[np.int32]] = []

    for i in feat_raw:
        x = np.array([int(j) for j in i], dtype=np.int32).reshape((9, 9, 1))
        feat_list.append(x)

    feat = np.array(feat_list, dtype=np.float32)
    feat = feat / 9.0
    feat -= 0.5

    for i in label_raw:
        x = np.array([int(j) for j in i], dtype=np.int64).reshape((81, 1)) - 1
        label_list.append(x)

    label = np.array(label_list, dtype=np.int64)

    x_train, x_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test
