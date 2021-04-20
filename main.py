import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 1.Loading data
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', None)

data = pd.read_csv('cakes_train.csv')
print(data)
print('First five elements')
print(data.head)
print('Last five elements')
print(data.tail())

# 2. Data analysis
print(data.info())
print(data.describe())

# dependency of eggs and type of cakes
data['eggs'] = data['eggs'].to_numpy().dot(63)
sb.displot(data, x='eggs', hue='type', hue_order=['cupcake', 'muffin'], multiple='fill', bins=8)
plt.xticks(rotation=90)
plt.show()

# dependency of flour and type of cakes
bins = data['flour'].nunique()
data['flour_chunks'] = pd.qcut(data['flour'], 10, duplicates='drop')
data = data.sort_values(by='flour_chunks')
data['flour_chunks'] = data['flour_chunks'].astype(str)
sb.displot(data, x='flour_chunks', hue='type', hue_order=['cupcake', 'muffin'], multiple='fill', bins=10)
plt.xticks(rotation=0)
plt.show()

# dependency of baking powder and type of cakes
bins = data['baking_powder'].nunique()
data['chunk_baking_powder'] = pd.qcut(data['baking_powder'], 10, duplicates='drop')
data = data.sort_values(by='chunk_baking_powder')
data['chunk_baking_powder'] = data['chunk_baking_powder'].astype(str)
sb.displot(data, x='baking_powder', hue='type', hue_order=['cupcake', 'muffin'], multiple='fill', bins=10)
plt.xticks(rotation=90)
plt.show()

# dependency of sugar and type of cakes
bins = data['sugar'].nunique()
data['sugar_chunk'] = pd.qcut(data['sugar'], 20, duplicates='drop')
data = data.sort_values(by='sugar_chunk')
data['sugar_chunk'] = data['sugar_chunk'].astype(str)
sb.displot(data, x='sugar_chunk', hue='type', hue_order=['cupcake', 'muffin'], multiple='fill', bins=20)
plt.xticks(rotation=0)
plt.show()

# dependency of milk and type of cakes
bins = data['milk'].nunique()
data['milk_chunk'] = pd.qcut(data['milk'], 20, duplicates='drop')
data = data.sort_values(by='milk_chunk')
data['milk_chunk'] = data['milk_chunk'].astype(str)
sb.displot(data, x='milk_chunk', hue='type', hue_order=['cupcake', 'muffin'], multiple='fill', bins=10)
plt.xticks(rotation=0)
plt.show()

# dependency of butter and type of cakes
bins = data['butter'].nunique()
data['butter_chunk'] = pd.qcut(data['butter'], 20, duplicates='drop')
data = data.sort_values(by='butter_chunk')
data['butter_chunk'] = data['butter_chunk'].astype(str)
sb.displot(data, x='butter_chunk', hue='type', hue_order=['cupcake', 'muffin'], multiple='fill', bins=10)
plt.xticks(rotation=0)
plt.show()

le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

sb.pointplot(x='eggs', y='type', data=data)
plt.show()

print(data)
# 3. Data cleansing

# 4. Data featuring
data = data.drop(columns=['flour_chunks'])
data = data.drop(columns=['chunk_baking_powder'])
data = data.drop(columns=['sugar_chunk'])
data = data.drop(columns=['milk_chunk'])
data = data.drop(columns=['butter_chunk'])
# deleting flour column because it doesn't say us to much
data = data.drop(columns=['flour'])

# group column eggs based on graphic
data.loc[(data['eggs'] <= 100), 'eggs'] = 0
data.loc[(data['eggs'] > 100) & (data['eggs'] <= 150), 'eggs'] = 1
data.loc[(data['eggs'] > 150) & (data['eggs'] <= 300), 'eggs'] = 2
data.loc[(data['eggs'] > 300) & (data['eggs'] <= 400), 'eggs'] = 3
data.loc[(data['eggs'] > 400), 'eggs'] = 4

# group column baking_powder based on graphic
data.loc[(data['baking_powder'] < 2), 'baking_powder'] = 0
data.loc[(data['baking_powder'] >= 2) & (data['baking_powder'] <= 5), 'baking_powder'] = 1
data.loc[(data['baking_powder'] > 5) & (data['baking_powder'] <= 6), 'baking_powder'] = 2
data.loc[(data['baking_powder'] > 6) & (data['baking_powder'] <= 8), 'baking_powder'] = 3
data.loc[(data['baking_powder'] > 8) & (data['baking_powder'] <= 11), 'baking_powder'] = 4
data.loc[(data['baking_powder'] > 11), 'baking_powder'] = 5

# group column sugar based on graphic
data.loc[(data['sugar'] < 50), 'sugar'] = 0
data.loc[(data['sugar'] >= 50) & (data['sugar'] <= 171), 'sugar'] = 1
data.loc[(data['sugar'] > 171) & (data['sugar'] <= 250), 'sugar'] = 2
data.loc[(data['sugar'] > 250) & (data['sugar'] <= 267), 'sugar'] = 3
data.loc[(data['sugar'] > 267) & (data['eggs'] <= 500), 'sugar'] = 4
data.loc[(data['sugar'] > 500) & (data['eggs'] <= 1000), 'sugar'] = 5
data.loc[(data['sugar'] > 1000), 'sugar'] = 6

# group column milk based on graphic
data.loc[(data['milk'] < 10), 'milk'] = 0
data.loc[(data['milk'] >= 10) & (data['milk'] <= 59), 'milk'] = 1
data.loc[(data['milk'] > 59) & (data['milk'] <= 120), 'milk'] = 2
data.loc[(data['milk'] > 120) & (data['milk'] <= 151), 'milk'] = 3
data.loc[(data['milk'] > 151) & (data['milk'] <= 300), 'milk'] = 4
data.loc[(data['milk'] > 300), 'milk'] = 5

# group column butter based on graphic
data.loc[(data['butter'] < 14), 'butter'] = 0
data.loc[(data['butter'] >= 14) & (data['butter'] <= 28), 'butter'] = 1
data.loc[(data['butter'] > 28) & (data['butter'] <= 76), 'butter'] = 2
data.loc[(data['butter'] > 76) & (data['butter'] <= 114), 'butter'] = 3
data.loc[(data['butter'] > 114) & (data['butter'] <= 142), 'butter'] = 4
data.loc[(data['butter'] > 142) & (data['butter'] <= 151), 'butter'] = 5
data.loc[(data['butter'] > 151) & (data['butter'] <= 207), 'butter'] = 6
data.loc[(data['butter'] > 207) & (data['butter'] <= 284), 'butter'] = 7
data.loc[(data['butter'] > 284), 'butter'] = 8

print(data)

# 5. Model Training
dtc_model = DecisionTreeClassifier(criterion='entropy')
X = data.drop(columns=['type'])
y = data['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=234, shuffle=True)
dtc_model.fit(X_train, y_train)

labels_predicted = dtc_model.predict(X_test)
ser_pred = pd.Series(data=labels_predicted, name='Predicted', index=X_test.index)
res_df = pd.concat([X_test, y_test, ser_pred], axis=1)
print(res_df.head(10))

# calculating accuracy
print(f'Model score: {dtc_model.score(X_test, y_test):0.3f}')

cv_res = cross_validate(dtc_model, X, y, cv=10)
print(sorted(cv_res['test_score']))
print(f'CV score: {cv_res["test_score"].mean():0.3f}')

# 6.Visualisation of the tree
fig, axes = plt.subplots(1, 1, figsize=(8, 3), dpi=400)

tree.plot_tree(decision_tree=dtc_model, max_depth=3,
               feature_names=X.columns, class_names=['cupcake', 'muffin'],
               fontsize=3, filled=True)
plt.show()