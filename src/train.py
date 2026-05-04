import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_random_forest(x_train, y_train, save_path='models/random_forest.pkl'):
    # """
    # Random Foresttttttttttttt
    print("start training estimate around 67s")
    rndForest_clsf = RandomForestClassifier()

    # n_estimators=100 => build 100 trees
    param_grid = [
        {"n_estimators": [100, 200, 300, 400]}
    ]

    grid_search = GridSearchCV(estimator=rndForest_clsf, param_grid=param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(x_train, y_train)

    final_model = grid_search.best_estimator_

    # save model:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(final_model, save_path)
    print("model is saved")

    return final_model

    print("best parameter:", grid_search.best_params_)


def load_model(path):
    # load a previously saved model
    return joblib.load(path)





# if __name__ == '__main__':
#     import pandas as pd

#     print("Loading processed data...")
#     X_train = pd.read_csv()
#     y_train = pd.read_csv()

#     model = train(x_train, y_train)
