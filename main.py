import numpy as np
from sklearn.metrics import confusion_matrix

from src.data_loader.data_loader import load_data
from src.plot.plotting import plot_col_hist, analyze_correlation
from src.metrics.calculate_metrics import calculate_metrics
from src.train.train import get_train_test_set, calculate_weights, train_classifier


# v2 = data.iloc[:,2]
# v7 = data.iloc[:,7]
# v29 = data.iloc[:,29]

# print(v2.shape)

# plt.figure(figsize=(8,8))
# plt.scatter(x=v2, y=v7)
# plt.show()

# plt.figure(figsize=(8,8))
# plt.scatter(x=v29, y=v7)
# plt.show()

# plt.figure(figsize=(8,8))
# plt.scatter(x=v2, y=v29)
# plt.show()





[data, n, m] = load_data('data/creditcard_modify.csv')

X_train, X_test, y_train, y_test = get_train_test_set(data)

cw, sw = calculate_weights(y_train)

classifier = train_classifier(X_train, y_train)
classifier_balanced = train_classifier(X_train, y_train, sw)

y_pred = classifier.predict(X_test)
y_pred_balanced = classifier_balanced.predict(X_test)

print(type(y_pred))

   

e = np.mean((y_pred - y_test)**2)
print("e=", e)

e_balanced = np.mean((y_pred_balanced - y_test)**2)
print("e_balanced=", e_balanced)

M = confusion_matrix(y_test, y_pred)
print("M=", M)

M_balanced = confusion_matrix(y_test, y_pred_balanced)
print("M_balanced=", M_balanced)

calculate_metrics(M)
calculate_metrics(M_balanced)