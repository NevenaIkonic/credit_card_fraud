import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

data = pd.read_csv('data/creditcard_modify.csv')

# n = data.shape
# print(n)

# rows = n[0]
# cols = n[1]

# print(data.head())
# print(data.columns)

# print(data.dtypes)


data_numeric = data.apply(pd.to_numeric, errors='coerce')

clean_data = data_numeric.dropna()
#print(clean_data)

#non_numeric_rows = data_numeric[data_numeric.isna().any(axis=1)]
#print(non_numeric_rows)



# data['Time'].plot(kind='hist', bins=100, figsize=(14, 6))
# plt.show()


# for i in range (1,29):
#     data[f'V{i}'].plot(kind='hist', bins=100, figsize=(14,6))
#     plt.show()

# data['Amount'].plot(kind='hist', bins=100, figsize=(14, 6))
# plt.show()

# data['Class'].plot(kind='hist', bins=100, figsize=(14, 6))
# plt.show()

# correlation_matrix = data.corr()
# #print(correlation_matrix)

# plt.figure(figsize=(10,10))
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.savefig('plots/img.png')
# plt.show()

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


x = clean_data.drop('Class', axis=1)
y = clean_data['Class']

# print(x.shape)
# print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

classifier = LogisticRegression()

cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
print(cw)
sw = compute_sample_weight('balanced', y=y_train)
classifier.fit(X_train, y_train, sample_weight=sw)

y_pred = classifier.predict(X_test)

print(type(y_pred))

e = np.mean((y_pred - y_test)**2)
print("e=", e)

print(np.mean(y_test))