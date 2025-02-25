import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('data/creditcard_modify.csv')

# n = data.shape
# print(n)

# rows = n[0]
# cols = n[1]

# print(data.head())
# #print(data.columns)

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

correlation_matrix = data.corr()
#print(correlation_matrix)

plt.figure(figsize=(6,4))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

v2 = data.iloc[:,2]
v7 = data.iloc[:,7]
v29 = data.iloc[:,29]

print(v2.shape)

plt.figure(figsize=(8,8))
plt.scatter(x=v2, y=v7)
plt.show()

plt.figure(figsize=(8,8))
plt.scatter(x=v29, y=v7)
plt.show()

plt.figure(figsize=(8,8))
plt.scatter(x=v2, y=v29)
plt.show()
