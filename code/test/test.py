import numpy as np
import pandas as pd
import streamdm as sdm
from scipy.io import arff
from sklearn.model_selection import train_test_split


with open("data.arff") as f:
    learner = sdm.HoeffdingTree()
    print(f'Method: {learner}')

    data, meta = arff.loadarff(f)
    print(meta)
    print(f'\nDataset {data.shape}: \n{data}')

    set_size = 1000
    df = pd.DataFrame(data[0:set_size])
    X = df.iloc[:, 0:13].to_numpy()
    y = df.iloc[:, 13].to_numpy().astype(int)
    print(f'\nX {X.shape}: \n{X}')
    print(f'\ny {y.shape}: \n{y}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    print(f'\nX_train {X_train.shape}: \n{X_train}')
    print(f'\ny_train {y_train.shape}: \n{y_train}')
    print(f'\nX_test {X_test.shape}: \n{X_test}')
    print(f'\ny_test {y_test.shape}: \n{y_test}')

    learner.fit(X_train, y_train)
    predictions = learner.predict(X_test)
    print(f'\npredictions {predictions.shape}: \n{predictions}')
    print('Finished!')
