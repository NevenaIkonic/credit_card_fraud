import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.linear_model import LogisticRegression



def get_train_test_set(data:pd.DataFrame) -> list:
    '''
    Creates training and test set
    '''
    x = data.drop('Class', axis=1)
    y = data['Class']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test



def calculate_weights(y_train):
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    sw = compute_sample_weight('balanced', y=y_train)

    return cw, sw



def train_classifier(X_train, y_train, sw=None):
    if sw is None:
        sw = np.ones_like(y_train)

    classifier = LogisticRegression()

    classifier.fit(X_train, y_train, sample_weight=sw)

    return classifier
