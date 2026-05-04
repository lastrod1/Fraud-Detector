import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys


def load_data(train_path, test_path):
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"train shape: {train.shape}")
    print(f"test shape: {test.shape}")
    return train, test


def preprocess(train_df, test_df):
    # separate labels
    y_train = train_df['is_fraud']
    y_test = test_df['is_fraud']

    # drop columns we dont need
    cols_to_drop = [
        'trans_date_trans_time', 'cc_num', 'first', 'last',
        'street', 'trans_num', 'unix_time', 'dob',
        'merchant', 'zip', 'city', 'state',
        'job'   #67
    ]
    train_df = train_df.drop(columns=cols_to_drop)
    test_df = test_df.drop(columns=cols_to_drop)

    # convert gender to numbers
    # M = 0, F = 1
    # M = 0, F = 1
    train_df['gender'] = train_df['gender'].map({'M': 0, 'F': 1})
    test_df['gender'] = test_df['gender'].map({'M': 0, 'F': 1})


    # category column has text like 'gas_transport', 'grocery_pos' etc
    # convert each category into its own 0 or 1 column
    train_df = pd.get_dummies(train_df, columns=['category'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['category'], drop_first=True)

    # remove the label column from features
    X_train = train_df.drop(columns=['is_fraud'])
    X_test = test_df.drop(columns=['is_fraud'])

    # after get_dummies, train and test might have different columns
    # for example train might have category_gas but test might not
    # we need to make sure they have the same columns
    train_columns = list(X_train.columns)
    test_columns = list(X_test.columns)

    missing_in_test = []
    for col in train_columns:
        if col not in test_columns:
            missing_in_test.append(col)

    for col in missing_in_test:
        X_test[col] = 0

    X_test = X_test[train_columns]

    # scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("preprocess done!")

    return X_train, X_test, y_train.values, y_test.values