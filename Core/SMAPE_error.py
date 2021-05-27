import numpy as np


def SMAPE_error(y_pred, y_true):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)
    if np.any(y_true == 0) and np.any(y_pred == 0):
        print("Found zeroes in y_true and y_pred. SMAPE undefined. Removing from set...")
        idx = np.where(y_true == 0)
        idy = np.where(y_pred == 0)
        y_true = np.delete(y_true, [idx, idy])
        y_pred = np.delete(y_pred, [idx, idy])

    smape = (200/y_true.shape[0]) * np.sum((np.abs((y_true - y_pred)) / (np.abs(y_true) + np.abs(y_pred))))

    if smape > 17:
        return np.array(0)
    else:
        return 200 * (((17 - smape) / 17) ** 0.35)
