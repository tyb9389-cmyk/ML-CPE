import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split dataset into train / validation / test sets while preserving class distribution.

    Args:
        X: Feature array or list.
        y: Labels array or list.
        test_size: Proportion of the dataset to include in the test split.
        val_size: Proportion of the entire dataset to include in the validation split.
        random_state: Controls the shuffling applied to the data before applying the split.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # 1. Split off the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 2. Adjust validation ratio relative to the remaining training data
    val_ratio = val_size / (1.0 - test_size)

    # 3. Split off the validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test